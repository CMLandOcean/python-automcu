#!/usr/bin/env python
import os
import argparse
import tqdm

import rasterio
import numpy as np
import pandas as pd
import spectral

sample_axis = 0
band_axis = 1

def bandwind_from_wls(centers, wl1, wl2):
    """Given a list of band centers and a pair of floats
       defining a range of wavelengths, find the bands that are covered by this
       range (but not outside). Bands are 0-based, so first band is 0"""
    ##First check if there's overlap, return None if not
    if (wl1 > centers[-1]) or (wl2 < centers[0]):
        return None
    minband = len(centers) + 1
    maxband = -1 
    for ci, c in enumerate(centers):
        if (c > wl1) and (c < wl2):
            if ci < minband:
                minband = ci
            if ci > maxband:
                maxband = ci
    if (maxband < 0) or (minband > len(centers)):
        ##Should not get here if there's overlap, unless wl range is tiny
        return None
    return [minband,maxband]

class AutoMCU(object):
    """Class to setup and run AutoMCU according to specify configuration
       Inputs:
       em_csvs           : List of csv files that contain endmember spectra, one
                           for each class - typically PV, NPV, and Bare. CSVs
                           should be arranged such that samples are columns, and
                           bands are rows, with the first column containing the
                           band center wavelengths in nanometers. At least two
                           classes must be specified.
       em_counts=[]      : Either a list of integers specifying how many em
                           samples to collect from each em class, a single
                           integer specifying how many samples to take from all
                           classes, or an empty list (or equivalently None). In
                           the first case, the length of this list must match
                           the length of em_csvs. In the first and second case,
                           the integer 0 means all available samples. An empty
                           list or None is equivalent to [0]*len(em_csvs).
       wl_ranges=None    : A list or tuple of 2-element lists or tuples of
                           floats, i.e. [(600.0,750),(2030,2380)], that will be
                           used to identify the band ranges used for unmixing. 
                           Data from the bands specified by these wavelengths 
                           will be lumped into a single array for fitting.
                           A minimum of one band window must be specified.
       band_ranges=None  : Similar to band_ranges, but specifying starting and
                           ending integer band indexes (first band is band 0)
                           for whatever image this is applied on.
       tied=True         : Reflectance spectra from each band range are 
                           subtracted from the first value in the window to
                           help relieve brightness issues. (First value in
                           unmixing array becomes 0.0 for each windows start)
       divide=False      : Similar to tied, but window values are dividied by
                           the first entry. (First value in each window will
                           become 1.0)
       shade=False       : Add an endmember for shade, which is constant 0.0
       sum_to_one=True   : Add an element with value 1.0 to the end of the
                           unmixing array and the input refl to help coerce sums
                           of the endmember fractions to be near 1.0.
       iterations=30     : How many MC runs to do for each pixel.
       outscale=1000     : Scale value applied to unmixing results to make them
                           fit into data type specified for output.
       nodata=-9999      : Output no data value written into metadata. If None, 
                           missing data in output will be 0, but this will not
                           be recorded in metadata.
       dtype="int16"     : Rasterio-usable specification of output map data type.
       otype="GTiff"     : GDAL shortname for output data format.
       co=[]             : GDAL data format writing options.
       emfwhm=False      : Assume second column of em csvs are fwhm values
       nointerp=False    : Assume em csvs have same count and wavelengths as
                           input reflectance data (will be checked)
       match_c           : Recreate a bug in original C code that shifted input
                           reflectance data by one index/band
    """

    def __init__(self, em_csvs, em_counts=[], em_names=[],
                 band_ranges=None, wl_ranges=None, tied=True, divide=False,
                 shade=False, soil_frac=False, sum_to_one=True, iterations=30,
                 outscale=1000, nodata=-9999, dtype="int16", otype="GTiff",
                 co=[], emfwhm=False, nointerp=False, match_c=False):
        self.tied = tied
        self.divide = divide
        self.shade = shade
        self.soilfrac = soil_frac
        self.sum1 = sum_to_one
        self.niter = iterations  # defined by user
        self.outscale = outscale
        self.nodata = nodata
        self.dtype = dtype
        self.otype = otype
        self.dointerp = not nointerp
        self.match_c = match_c
        self.dopt = dict([s.split("=") for s in co])

        ##Filled later
        self._ems_interp = None
        self._ems_proc = None
        self._em_avg = None

        ##Check the endmember args
        ##########################
        # em_csv should be a list
        assert isinstance(em_csvs,list)
        # For now, lets say em classes should be 2 or more
        assert len(em_csvs) >= 2
        # Counts should either be an empty, a single integer, or a list of
        # integers the same size as em_csvs
        if not em_counts:
            em_counts = [0]*len(em_csvs)
        elif isinstance(em_counts,list):
            assert len(em_csvs) == len(em_counts)
        else:
            em_counts = [em_counts]*len(em_csvs)
        # Check that files exist
        for csv in em_csvs:
            assert os.path.exists(csv)
        self._ems_raw = []
        self._ems_wl = []
        self._ems_fwhm = []
        for csv, count in zip(em_csvs, em_counts):
            ##Here csv should be formatted with a header row and samples listed
            ## in columns, bands as rows. Wl should be the first columns, and 
            ## if --emfwhm is specified, fwhm should be column 2
            tmpdf = pd.read_csv(csv)
            self._ems_wl.append(tmpdf.iloc[:,0].to_numpy())
            st = 1 
            if emfwhm: 
                self._ems_fwhm.append(tmpdf.iloc[:,1].to_numpy())
                st = 2
            else:
                self._ems_fwhm.append(None)
            if count > 0:
                maxcount = tmpdf.shape[1] - st
                if count > maxcount:
                    count = maxcount
                end = st + count
                print(f"Reading {count} samples from {csv}")
                self._ems_raw.append(tmpdf.iloc[:,st:end].to_numpy().T)
            else:
                print(f"Reading all samples of {csv}")
                self._ems_raw.append(tmpdf.iloc[:,st:].to_numpy().T.T)

        #Save the number of em classes 
        self.num_class = len(self._ems_raw)

        ##Check for em_names 
        if em_names:
            self._em_names = em_names
        else:
            self._em_names = [f"Class{n:d}" for n in range(1,self.num_class+1)]

        if self.shade:
            self.num_class += 1
            self._em_names.append("Shade")

        print("Classes will be:")
        for c in self._em_names:
            print(f"{c}")

        ##Check the specified ranges
        ############################
        # One and only one should be not None
        if band_ranges is not None:
            # There should be at least one window
            self._wl_ranges = None
            assert isinstance(band_ranges,list) or isinstance(band_ranges,tuple)
            assert len(band_ranges) > 0
            # If the first element is a float, then assume only one range
            if not (isinstance(band_ranges[0],list) or
                    isinstance(band_ranges[0],tuple)):
                self._band_ranges = [band_ranges[:2]]
            else:
                self._band_ranges = [list(p[:2]) for p in band_ranges]
            # Test that all range entries can be floats
            for pr in self._band_ranges:
                for it in pr:
                    try:
                        int(it)
                    except ValueError:
                        raise RuntimeException(f"Window range entry {it} could not be converted to int")
                    except TypeError:
                        raise RuntimeException(f"Invalid window range entry: {it}")
            pass
        elif wl_ranges is not None:
            # There should be at least one window
            self._band_ranges = None
            assert isinstance(wl_ranges,list) or isinstance(wl_ranges,tuple)
            assert len(wl_ranges) > 0
            # If the first element is a float, then assume only one range
            if not (isinstance(wl_ranges[0],list) or
                    isinstance(wl_ranges[0],tuple)):
                self._wl_ranges = [wl_ranges[:2]]
            else:
                self._wl_ranges = [list(p[:2]) for p in wl_ranges]
            # Test that all range entries can be floats
            for pr in self._wl_ranges:
                for it in pr:
                    try:
                        float(it)
                    except ValueError:
                        raise RuntimeException(f"Window range entry {it} could not be converted to float")
                    except TypeError:
                        raise RuntimeException(f"Invalid window range entry: {it}")
        else:
            raise ArgumentError("One of wl_ranges or band_ranges must be specified")
        return 

    def _find_hdr_name(self,filepath):
        inhdr = None
        ##Go through common hdr name practices looking for one that exists
        inhdropts = [
            os.path.splitext(filepath)[0]+".hdr",
            filepath+".hdr"
        ]
        for opt in inhdropts:
            if os.path.exists(opt):
                print(f"Found input hdr file at {opt}")
                inhdr = opt
                break
        ##Will be None if no options exist
        return inhdr

    def interpolate_ems(self, wl, fwhm=None):
        if fwhm is None:
            avg_spacing = np.diff(self.wl).mean()
            fwhm = [avg_spacing]*len(wl)
        self._ems_interp = []
        for rawmat, rawwl, rawfwhm in \
                zip(self._ems_raw, self._ems_wl, self.ems_fwhm):
            ##Build fwhm if needed
            if rawfwhm is None:
                avg_spacing = np.diff(rawwl).mean()
                rawfwhm = [avg_spacing]*len(rawwl)
            ##Interpolate 
            brs = spectral.BandResampler(rawwl, wl, rawfwhm, fwhm)
            self._ems_interp.append(np.concatenate(
                [brs(v) for v in rawmat ], axis=sample_axis))
        return

    def average_interpolated_ems(self):
        if not self._ems_interp:
            raise RuntimeError(
                    "interpolated_ems() has not been called yet")
        use_bands = []
        for rb in self.regdefs:
            use_bands.extend(rb)
        self._em_avg = np.concatenate([emarr[:,use_bands].mean(
                                     axis=sample_axis,keepdims=True) \
                                         for emarr in self._ems_interp],
                                 axis = sample_axis)
        return

    def make_unmixing_array(self,arr,match_c=False):
        """Select correct bands from arr, a NxB array, where n is number of
           samples and B is number of bands, and perform tying or dividing if
           requested"""
        dat_list = [arr[:,bandlist] for bandlist in self.regdefs]

        if self.divide:
            for dat in dat_list:
                dat /= dat[:,0][:,np.newaxis]
        elif self.tied:
            for dat in dat_list:
                dat -= dat[:,0][:,np.newaxis]
        
        dat = np.concatenate(dat_list, axis=band_axis)
        ##Bug in automcu.c makes Y(a.k.a., b) data shifted by one index
        ## This results in first band being omitted, and lastband being 0, since
        ##  fsubimgslice array is 0 after last svd band:
        ##   /* This is a bug since fsubimgslice is 0-based indexing, and it  */
        ##   /*     should be fsubimgslice[i][k-1] */
        ##   for (k=1; k<=num_prim_imgbands; k++) imgspec[k] = fsubimgslice[i][k];
        ##   
        ##   if (sum_to_one == TRUE) imgspec[num_prim_svdbands] = 1.0; 
        ## Re-enact this bug if requested
        if match_c:
            dat = np.roll(dat,-1,axis=band_axis)
            dat[:,-1] = 0.0

        if self.sum1:
            return np.concatenate([dat, 
                                   np.full((dat.shape[0],1),1.0)],
                                       axis=band_axis)
        else:
            return dat
        
    def process_iterpolated_ems(self):
        """With em data now matching application bands, process for unmixing"""
        if self._ems_interp is None:
            raise RuntimeError("interpolate_ems() has not been called yet")

        self._ems_proc = []
        for dat_i, dat in enumerate(self._ems_interp):
            procdat = self.make_unmixing_array(dat)
            ##BandResampler returns NaN for wl with no overlap
            if np.any(~np.isfinite(procdat)):
                raise RuntimeError(
                    "Processed em data contains non-finite values")
            self._ems_proc.append(procdat)
        return

    def _random_unmixing_libs(self,n):
        picks = np.concatenate([
                s[np.random.choice(s.shape[0],size=n,replace=True),:][:,np.newaxis,:]
                            for s in self._ems_proc],axis=band_axis)
        return picks # n x num_class x num_bands

    def unmix_array(self, arr):
        """Actually do the unmixing for an array of valid refl data of shape
           N x B, where N is number of samples and B is number of raw spectral
           bands"""

        ##Will use pseudoinverse to linearly solve Y = X*coef
        ##Done for all samples simultaneously thanks to numpy magic

        ##Check that some members are not None
        if self._ems_proc is None:
            raise RuntimeError(
                "process_interpolated_ems() has not been called yet")

        if self._em_avg is None:
            raise RuntimeError(
                "average_interpolated_ems() has not been called yet")

        #########################
        ##Prepare Y array (N x B)
        #########################

        ##Process data in to unmixing array by applying tieing/dividing/sum1
        Y = self.make_unmixing_array(arr,self.match_c)
        num_samples = Y.shape[sample_axis]
        num_bands = Y.shape[band_axis]
        #Y is shape (num_samples, num_bands)


        ##Also get original refl spectrum for the selected bands
        ##Shape (num_samples, num_bands)
        Yorig = np.concatenate([
                  arr[:,bandlist] for bandlist in self.regdefs],
                  axis=band_axis)

        ######################################################
        ##Prepare X matrices from randomly selected endmembers
        ######################################################

        ##Collect needed number of random draws from SPECs
        ## One randomly-selected sample from each class for each iteration for
        ##    each sample
        ##shape (num_samples, self.niter, num_class, num_bands)
        ######################################################################
        random_selections = self._random_unmixing_libs(
                self.niter*num_samples).reshape(num_samples,
                                                self.niter,
                                                self.num_class,
                                                num_bands)

        #################
        ##Do the unmixing
        #################

        ##Collect pseudo_inverses into a new array 
        #shape (num_samples, self.niter, num_bands, num_class)
        ####################################################
        pinvs = np.linalg.pinv(random_selections)

        ##Get optimal coefficients for each sample and iteration
        ########################################################
        # shape num_samples, self.niter, num_class)
        ########################################################
        coefs = np.concatenate([
            np.dot(pinvs[c,i,:].T,Y[c,:])[np.newaxis,:]
                        for c in range(num_samples) 
                            for i in range(self.niter)],axis=sample_axis)
        coefs = coefs.reshape(num_samples,self.niter,self.num_class)

        #######################
        ##Compute trimmed stats
        #######################

        ##Compute quantiles across just iterations
        ## Each is shape (num_samples, num_class)
        ##########################################
        q10, q90 = np.percentile(coefs,[10,90],axis=1)    

        ##Trim 10% of coefs from each end by sample and class
        #####################################################
        coefs[np.logical_or(
            coefs < q10[:,np.newaxis,:],
            coefs > q90[:,np.newaxis,:])] = np.nan

        ##Compute mean and stdev of middle 80%
        ## each shape (num_samples, num_class)
        ######################################
        means = np.nanmean(coefs,axis=band_axis) # num_samples x num_class
        sds = np.nanstd(coefs,axis=band_axis) # num_samples x num_class

        ######################################################################
        ##Remix expected spectra for each sample and compute sample-level RMSE
        ######################################################################

        ##Build mixed spectra (in original refl scale for selected bands)
        ##(num_samples, num_class) ⋅ (num_class, num_bands) = 
        ##  (num_samples, num_bands)
        ##  Does not include band of 1s from sum1
        #################################################################
        preds = np.dot(means,self._em_avg)

        ##RMSE
        ## Shape (num_samples)
        ######
        rmses = np.sqrt(np.sum((preds - Yorig)**2,
                               axis=band_axis,
                               keepdims=True) /num_bands)


        return np.concatenate([means, sds, rmses], axis=band_axis)

    def apply_image(self, refl_path, input_scale, output_path, 
                  hdrpath=None, nblocks=0):

        ##Get info from input image
        ###########################

        ## Grab shape and metadata
        with rasterio.open(refl_path) as inref:
            imgshape = inref.shape
            imgoffset = (0,0)
            outmeta = inref.meta.copy()
            input_blocks = list(inref.block_windows())

        ## Modify if we're not using all blocks
        if nblocks > 0:
            input_blocks = input_blocks[:nblocks]
            minr, minc = imgshape
            maxr, maxc = (0,0)
            for w_i, wind in input_blocks:
                minr = min(minr, wind.row_off)
                minc = min(minc, wind.col_off)
                maxr = max(maxr, wind.row_off + wind.height - 1)
                maxc = max(maxc, wind.col_off + wind.width - 1)
            imgshape = (maxr - minr + 1, maxc - minc + 1)
            imgoffset = (minr, minc)

        ##Try to find hdr if not specified        
        if not hdrpath:
            hdrpath = self._find_hdr_name(refl_path)
        if not hdrpath:
            raise RuntimeError("No matching hdr found")
        if not os.path.exists(hdrpath):
            raise RuntimeError(f"HDR {hdrpath} does not exist")
        input_hdr = spectral.envi.read_envi_header(hdrpath)

        ##Get wavelengths from header
        if "wavelength" in input_hdr:
            self.wl = [float(wl) for wl in input_hdr["wavelength"]]
        else:
            raise RuntimeError(f"Could not find wavelengths in hdr")
        ##Get fwhm from header or build
        if "fwhm" in input_hdr:
            self.fwhm = [float(hm) for hm in input_hdr["fwhm"]]
        else:
            ## If fwhm missing, just assume equivalent to center spacing
            avg_spacing = np.diff(self.wl).mean()
            self.fwhm = [avg_spacing]*len(self.wl)
    
        ##Check that self.wl and self.fwhm lengths match up
        if len(self.wl) != len(self.fwhm):
            raise RuntimeError(f"Number of hdr wavelengths {len(self.wl)} != "\
                    "number of fwhm {len(self.fwhm)}")
    
        ##Check that number of self.wl is the same as the number of bands
        num_image_bands = int(input_hdr["bands"])
        if len(self.wl) != int(input_hdr["bands"]):
            raise RuntimeError(f"Number of hdr wavelengths {len(self.wl)} != "\
                    "number of image bands {num_image_bands}")

        ##Figure out band ranges
        ########################

        ## Build _band_ranges if wl_ranges provided
        if self._wl_ranges is not None:
            self._band_ranges = []
            for wlr in self._wl_ranges:
                br = bandwind_from_wls(self.wl, *wlr[:2])
                if br is None:
                    raise RuntimeError("Could not get band indices from wl"\
                            f" range {wlr}")
                self._band_ranges.append(br)
                print(f"Computed band range {br} "\
                      f" ({self.wl[br[0]]},{self.wl[br[1]]})"\
                      f" from wavelength window {wlr}")

        ##Convert _band_ranges to lists of indices
        self.regdefs = []
        print(f"Found {len(self._band_ranges)} band regions:")
        for br_i, br in enumerate(self._band_ranges):
            print(f"{br_i+1}: {br}:")
            self.regdefs.append(list(range(br[0],br[1]+1)))
            print(f"{self.regdefs[-1]}")

        num_selected_bands = sum([len(rd) for rd in self.regdefs])
        print(f"Total selected image bands: {num_selected_bands}")
        num_unmixing_bands = num_selected_bands
        if self.sum1:
            num_unmixing_bands += 1
            print(f"Total unmixing bands (incl. sum_to_one): {num_unmixing_bands}")

        ##Interpolate and process em libraries for use in building A matrices
        #####################################################################

        ##Build self._ems_interp
        ##Will create a list of three NxB matrices, each representing one of the
        ## endmember classes. N is the number of samples in the given class and
        ## B is the number of input image bands
        if self.dointerp:
            self.interpolate_ems(self.wl, self.fwhm)
        else:
            self._ems_interp = self._ems_raw

        ##Check that number of bands match
        for emnum, emdat in enumerate(self._ems_interp):
            if emdat.shape[band_axis] != len(self.wl):
                raise RuntimeError("Number of bands in interpolated em data"\
                                   f" for {self.em_names[emnum]}"\
                                   f" ({emdat.shape[band_axis]})"\
                                   " does not match number of image bands"\
                                   f" ({len(self.wl)})")

        ##Compute averaged endmembers (must be done before processing)
        ##############################################################
        self.average_interpolated_ems()

        ##Build self._ems_proc
        ##Will create a list of three NxB matrices, each representing one of the
        ## endmember classes. N is the number of samples in the given class and
        ## B is the number of unmixing bands (sum of band regions + any extra
        ## for self.sum1 and self.shade)
        ########################################################################
        self.process_iterpolated_ems()

        ##Add a single-entry extra class for shade
        if self.shade:
            ##Shade is just a row of 0's
            tmpshade = np.full((1,num_unmixing_bands),0.0)
            ##Last entry still 1.0 if sum1
            if self.sum1:
                tmpshade[0,-1] = 1.0
            ##Append it to the _ems_proc list 
            self._ems_proc.append(tmpshade)
            ##Add an average shade to the _em_avg array
            self._em_avg = np.concatenate([
                self._em_avg, np.full((1,num_selected_bands),0.0)],
                  axis=sample_axis)

        ##Prepare output map
        ####################

        ##Update the output meta to match args
        outmeta["dtype"] = self.dtype
        outmeta["driver"] = self.otype
        if self.nodata:
            outmeta["nodata"] = self.nodata
        else:
            outmeta.pop("nodata",None)
        outmeta["count"] = 2*self.num_class+1

        ##Fix output width and height based on nblocks
        outmeta["height"] = imgshape[0]
        outmeta["width"] = imgshape[1]
        outmeta["transform"] = inref.window_transform(
            rasterio.windows.Window(*reversed(imgoffset),*reversed(imgshape)))
        output_blocks = [
            (w_i, rasterio.windows.Window(wind.col_off - imgoffset[1],
                                          wind.row_off - imgoffset[0],
                                          wind.width, wind.height)) \
                for w_i, wind in input_blocks]

        ##Run AutoMCU by image blocks and write to output
        #################################################
        ##Zip the input and output blocks so we can iterate over pairs
        comb_blocks = list(zip([w for _, w in input_blocks],
                               [w for _, w in output_blocks]))
        
        with rasterio.open(output_path,'w',**outmeta,**self.dopt) as oref:

            ##Update band descriptions
            ##########################
            banddesc = [*self._em_names]
            banddesc.extend([f"sd{cls}" for cls in self._em_names])
            banddesc.append("RMSE")
            oref.descriptions = banddesc

            with rasterio.open(refl_path) as inref:
                for inwind, outwind in tqdm.tqdm(
                        comb_blocks,
                        desc="Processing blocks",
                        total=len(comb_blocks)):
                    ###############################################
                    ##Read image data and filter out no data pixels
                    ###############################################

                    ##Make output shape tuple and count total block pixels
                    ######################################################
                    out_bands = 2*self.num_class+1
                    out_shape = (out_bands,inwind.height,inwind.width)
                    num_pix = inwind.height*inwind.width

                    ##Read the data for this block
                    ##############################
                    val_dat = inref.read(window=inwind).reshape(
                            inref.count,-1)/input_scale

                    ##Filter for pixels with constant values across all 
                    ## bands (i.e. no data)
                    ###################################################
                    isval = ~np.all( val_dat == val_dat[0,:][np.newaxis,:], 
                                      axis=sample_axis)
                    ##How many valid pixels in this block window
                    num_valid = isval.sum()

                    ##Drop the nodata pixels
                    ########################
                    val_dat = val_dat[:,isval]
                    ##Shape (image_bands, num_valid)

                    ###########################################################
                    ##Unmix the valid data
                    ##Returns an array of concatenated mean coefficients, stdev
                    ## coefficients, and rmse
                    ###########################################################
                    results = self.unmix_array(val_dat.T).T

                    ##########################
                    ##Prepare and write output
                    ##########################
                    out_dat = np.zeros((out_bands,num_pix),dtype=np.int16)
                    out_dat[:,isval] = results * self.outscale
                    oref.write(out_dat.reshape(out_shape),window=outwind)



def main():
    parser = argparse.ArgumentParser(description="Run AutoMCU on an image")
    parser.add_argument("--verbose","-v",action="store_true",
                        help="Verbose output")
    parser.add_argument("--config", "-c", default=None,
                        help="Get settings from JSON-formatted config file,"
                             " instead of needing to specify command line args")
    parser.add_argument("--input_hdr", default=None,
                        help="Specify name of input ENVI hdr in case it is not"\
                             " the standard path.hdr or splitext(path).hdr")
    parser.add_argument("--wl_range", "-w", default=[],action="append",
                        help="A 2-element list or tuple of floats, i.e."\
                             " (600.0,750), that will be used to identify"\
                             " the band ranges used for unmixing."\
                             " Data from the bands specified by"\
                             " these wavelengths will be lumped into a single"\
                             " array for fitting. A minimum of one wl window"\
                             " must be specified. Both wl_range and band_range"\
                             " cannot be specified")
    parser.add_argument("--band_range", "-b", default=[],action="append",
                        help="A 2-element list or tuple of integers, i.e."\
                             " (65,72), that will be used to identify"\
                             " the band ranges used for unmixing."\
                             " Band numbers are 0-based, so first band is 0."\
                             " Data from the bands specified by"\
                             " these windows will be lumped into a single"\
                             " array for fitting. A minimum of one band window"\
                             " must be specified. Both wl_range and band_range"\
                             " cannot be specified")
    parser.add_argument("--notie",dest="tied",action="store_false",
                        help="Reflectance spectra from each band range are"\
                             " subtracted from the first window value to"\
                             " help relieve brightness issues. (First value"\
                             " in unmixing array becomes 0.0 for each"\
                             " window's start). By default this is on. Use this"\
                             " flag to disable.")
    parser.add_argument("--divide", action="store_true",
                        help="Reflectance spectra from each band range are"\
                             " divided by the first window value to"\
                             " help relieve brightness issues. (First value"\
                             " in unmixing array becomes 1.0 for each"\
                             " window's start). By default this is off.")
    parser.add_argument("--shade", action="store_true",
                        help="Add an endmember for shade")
    parser.add_argument("--sum_to_one", "-1", action="store_true",
                        help="Add an element with value 1.0 to the end of the"\
                             " unmixing array and the input refl to help"\
                             " coerce sums of the endmember fractions to be"\
                             " near 1.0.")
    parser.add_argument("--iterations","--count", type=int, default=30,
                        help="How many MC runs to do for each pixel. "\
                             " Default 30")
    parser.add_argument("--num_blocks", type=int, default=0,
                        help="How many MC runs to do for each pixel. "\
                             " Default all")
    parser.add_argument("--outscale",type=float,default=1000,
                        help="Scale value applied to unmixing results to make"\
                             " them fit into data type specified for output.")
    parser.add_argument("--nodata",type=float,default=None,
                        help="Output no data value written into metadata. If"\
                             " default None, missing data in output will be 0,"\
                             " but this will not be recorded in metadata.")
    parser.add_argument("--scale",type=float,default=1.0,
                        help="Apply this scale to input data to get"\
                             " reflectance scaled from 0 to 1")
    parser.add_argument("--dtype","--of", default="int16",
                        help="Rasterio-usable specification of output map"\
                             " data type.")
    parser.add_argument("--otype","--ot",default="GTiff",
                        help="GDAL shortname for output data format.")
    parser.add_argument("--co","-o",default=[],action="append",
                        help="GDAL data format writing options.")
    parser.add_argument("--emfwhm",action="store_true",
                        help="Assume second column of em csvs are fwhm values."\
                             " By default, average inter-band spacing is used")
    parser.add_argument("--nointerp",action="store_true",
                        help="Assume em csvs have same count and wavelengths"\
                             " as input reflectance data (will be checked)")
    parser.add_argument("--match_c",action="store_true",
                        help="Recreate a bug in original C code that shifted"\
                             " input reflectance data by one index/band")
    parser.add_argument("--names",default=[],action="append",metavar="STR[,STR]",
                        help="Supply names for the classes defined by the"\
                             " given CSV files. Can be specified multiple"\
                             " times or using comma separated values")
    parser.add_argument("input",help="ENVI-formatted reflectance image")
    parser.add_argument("output",help="AutoMCU result map filename")
    parser.add_argument("emlist",nargs="+",metavar="CSV<:INT>",
                        help="CSV file with list of samples for a given"\
                             " endmember class. CSV should should be arranged"\
                             " such that samples are columns, and bands are"\
                             " rows, with the first column containing the band"\
                             " center wavelengths in nanometers. At least one"\
                             " class must be specified. If a : and integer n"\
                             " immediately follow the filename, then only the"
                              " first n samples will be read")
    args = parser.parse_args()
    # args =\
    #  parser.parse_args(["debugging_data/GAO20220411t125900p0000_iacorn_refl214_ort_sub",
                       #  "debugging_data/testpy_1.tif",
                       #  "debugging_data/vswir_2011_2012_pv_599x214_cao3_BbyN.csv:100",
                       #  "debugging_data/vswir_2011_2012_npv_50x214_cao3_BbyN.csv:50",
                       #  "debugging_data/vswir_2011_2012_bare_160x214_cao3_BbyN.csv:160",
                       #  "--names","PV,NPV,Bare","--scale","10000",
                       #  "--band_range","6,35","--band_range","(165,205)",
                       #  "--nointerp","--emfwhm"])


    ##Check that we have ems
    em_csvs = []
    em_counts = []
    for st in args.emlist:
        parts = st.strip().split(":")
        if len(parts) > 1:
            em_csvs.append(parts[0])
            if len(parts[1]) > 0:
                em_counts.append(int(parts[1]))
            else:
                em_counts.append(0)
        else:
            em_csvs.append(parts[0])
            em_counts.append(0)

    em_names = []
    if args.names:
        for n in args.names:
            parts = n.strip().split(",")
            em_names.extend(parts)


    ###########################
    ##Process band or wl ranges
    ###########################
    ##Check that we have band or wavelength windows
    if (len(args.band_range) > 0) and (len(args.wl_range) > 0):
        raise RuntimeError("Can't mix wl_range and band_range entries")

    ##Convert to floats or ints
    if args.band_range:
        band_ranges = []
        wl_ranges = None 
        for bstr in args.band_range:
            nobrackets = bstr.strip("[](){}")
            parts = nobrackets.split(",",maxsplit=2)
            if len(parts) < 2:
                raise RuntimeError("Could not convert band range to integers:"\
                                   f" {bstr}")
            band_ranges.append([int(p) for p in parts])

    elif args.wl_range:
        band_ranges = None
        wl_ranges = []
        for wstr in args.wl_range:
            nobrackets = wstr.strip("[](){}")
            parts = nobrackets.split(",",maxsplit=2)
            if len(parts) < 2:
                raise RuntimeError("Could not convert wl range to integers:"\
                                   f" {wstr}")
            wl_ranges.append([float(p) for p in parts])
    else:
        raise RuntimeError("Band or wavelength ranges must be specified with"\
                           "either --wl_range or --band_range")

    #############################
    ##Create the AutoMCU instance
    #############################
    amcu = AutoMCU(em_csvs, em_counts, em_names=em_names,
                 band_ranges=band_ranges, wl_ranges=wl_ranges,
                 tied=args.tied, divide=args.divide, shade=args.shade,
                 sum_to_one=args.sum_to_one, iterations=args.iterations,
                 outscale=args.outscale, nodata=args.nodata, 
                 dtype=args.dtype, otype=args.otype, co=args.co,
                 emfwhm=args.emfwhm, nointerp=args.nointerp, 
                 match_c=args.match_c)

    ##########################
    ##Run AutoMCU on the image
    ##########################
    amcu.apply_image(args.input, args.scale, args.output,
                     hdrpath=args.input_hdr, nblocks=args.num_blocks)

    return 

if __name__ == "__main__":
    main()
