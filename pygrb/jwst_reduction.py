import mastquery.jwst
import numpy as np
from astroquery.mast import Observations
import tqdm
import glob
import os

import yaml
import warnings
import time

import grizli
from grizli import utils, jwst_utils

jwst_utils.set_quiet_logging()
utils.set_warnings()

import astropy.io.fits as pyfits
import jwst.datamodels
import jwst

import mastquery.jwst

import msaexp
from msaexp import pipeline
import msaexp.utils as msautils
import msaexp.slit_combine

msautils.SFLAT_STRAIGHTEN = 3



class msaexp_drp:
	def __init__(self, prog_id):
		"""
		prog_id : int
					The JWST observational program id.
		"""
		self.prog_id = prog_id

	def query_data(self, download_dir=None, download=False, token='5a07da0fb7bf4bc4824dd342adff5328'):
		"""
		rate_dir : str
					The directory within which the data should be downloaded. 
					Usually should be e.g., 9223_Fujimoto/rate24_ext/, but in 
					this case can be anything.
		download : bool
					Boolean whether to download the data or just view the files available.
		token : str
					The MAST token associated with your profile, to download the data.
		"""
		Observations.login(token=token)

		masks = pipeline.query_program(self.prog_id,
									   download=False,
									   detectors=['nrs1','nrs2'],
									   extensions=['uncal','s2d','rate'])
		
		rate_files = []
		msa_files = []
		for entry in tqdm.tqdm(masks):
			if ('MIRROR' in entry['filter-pupil']):
				continue
			
			uri = entry['dataURI']
		
			rate_files.append(uri)
				
			msa = entry['msametfl']
			msa_file = f'mast:JWST/product/{msa}'
			msa_files.append(msa_file)
			
		rate_files = np.sort(np.unique(rate_files))
		msa_files = np.sort(np.unique(msa_files))
		all_files = np.concatenate([rate_files,msa_files])
		
		print('Files available for download:')
		for file in all_files:
			print(file)

		if download==True:
			assert download_dir is not None, 'Please specify or create directory for data download.'
			os.chdir(download_dir)
			for name in all_files:
				Observations.download_file(name)

		print('Data query complete.')


	def download_rate_files(self, files, token='5a07da0fb7bf4bc4824dd342adff5328'):
		"""
		files : list of strings
				Explicit file names to download. e.g., ['jw06368001001_03101_00002_nrs1_rate.fits']
		token : str
				The MAST token associated with your profile, to download the data.
		"""

		Observations.login(token=token)

		for file in files:
			name = f'mast:JWST/product/{file}'
			Observation.download_file(name)

		print('File download complete.')
			
	def preprocess_nirspec_file(self, rate_dir='rate24_ext',
								rate_files=["jw04233006001_03101_00002_nrs1_rate.fits"],
								outroot="rubies-egs61-vx",
								fixed_slit=None,
								context="jwst_1225.pmap",
								clean=False,
								extend_wavelengths=True,
								undo_flat=True,
								by_source=False,
								**kwargs):
	    """
	    Run preprocessing calibrations for a single NIRSpec exposure
	    """
	    from msaexp import pipeline_extended
	    from grizli import jwst_level1
	    from jwst.assign_wcs.util import NoDataOnDetectorError
	
	    # jwst_utils.set_crds_context()

	    os.chdir(WORKPATH)
		
		rate_files = np.sort(rate_files)
	    for rate_file in rate_files:

			## Set file variables and start log file
	    	file_prefix = rate_file.split("_rate")[0]
	    	key = f"{outroot}-{file_prefix}"

	    	_ORIG_LOGFILE = utils.LOGFILE
	    	_NEW_LOGFILE = os.path.join(WORKPATH, file_prefix + "_rate.log.txt")
	    	utils.LOGFILE = _NEW_LOGFILE
	
		    msg = f"""# {rate_file} {outroot}
		    			jwst version = {jwst.__version__}
						grizli version = {grizli.__version__}
						msaexp version = {msaexp.__version__}
		    		"""
	    	utils.log_comment(utils.LOGFILE, msg, verbose=True)
	
	    	##  Check file exists
	    	if not os.path.exists(rate_file):
	        	msg = f"{rate_file} does not exist..."
	        	utils.log_comment(utils.LOGFILE, msg, verbose=True)
	        	return 3
	
	   		if not fixed_slit:
	   		    with pyfits.open(rate_file) as im:
	   		        if "MSAMETFL" in im[0].header:
	   		            msametf = im[0].header["MSAMETFL"]
	   		            mastquery.utils.download_from_mast([msametf], overwrite=False)
	
	   		            msa = msaexp.msa.MSAMetafile(msametf)
	   		            msa.merge_source_ids()
	   		            msa.write(prefix="", overwrite=True)
	
	   		        else:
	   		            msametf = None
	   		else:
	   		    msametf = None
	   		    by_source = False
	
		    utils.log_comment(utils.LOGFILE, "Reset DQ=4 flags", verbose=True)
	        with pyfits.open(rate_file, mode="update") as im:
	            im["DQ"].data -= im["DQ"].data & 4
	            im.flush()
	

	    # Split into groups of 3 exposures
	    groups = pipeline.exposure_groups(files=rate_files, split_groups=True)
	    print("Files:")
	    print("======")
	    print(yaml.dump(dict(groups)))
	
	    # Single exposure groups
	    single_exposure_groups = {}
	    for g in groups:
	        for exp, k in zip(
	            groups[g], "abcdefghijklmnopqrstuvwxyz"[: len(groups[g])]
	        ):
	            gr = g.replace("-f", f"{k}-f").replace("-clear", f"{k}-clear")
	            single_exposure_groups[gr] = [exp]
	    print(yaml.dump(dict(single_exposure_groups)))
	
	    
	
	    
	
	    source_ids = None
	    pad = 0
	    sources = None
	    for g in groups:
	        for exp, k in zip(
	            groups[g], "abcdefghijklmnopqrstuvwxyz"[: len(groups[g])]
	        ):
	            mode = g.replace("-f", f"{k}-f").replace("-clear", f"{k}-clear")
	            xmode = f"{mode}-fixed" if fixed_slit else mode
	
	            if sources is not None:
	                source_ids = sources[g]  # [3:6]
	                if len(source_ids) < 1:
	                    source_ids = None
	            else:
	                source_ids = None
	
	            if os.path.exists(f"{xmode}.start"):
	                print(f"Already started: {mode}")
	                continue
	
	            source_ids = None
	            positive = False
	
	            if not os.path.exists(f"{xmode}.slits.yaml"):  #
	                with open(f"{xmode}.start", "w") as fp:
	                    fp.write(time.ctime())
	
	                if 0:
	                    source_ids = sources[mode]
	
	                if fixed_slit:
	                    for _file in single_exposure_groups[mode]:
	                        with pyfits.open(_file, mode="update") as _im:
	                            ORIG_EXPTYPE = _im[0].header["EXP_TYPE"]
	                            if ORIG_EXPTYPE != "NRS_FIXEDSLIT":
	                                print(f"Set {_file} MSA > FIXEDSLIT keywords")
	                                _im[0].header["EXP_TYPE"] = "NRS_FIXEDSLIT"
	                                _im[0].header[
	                                    "APERNAME"
	                                ] = f"NRS_{fixed_slit}_SLIT"
	                                _im[0].header["OPMODE"] = "FIXEDSLIT"
	                                _im[0].header["FXD_SLIT"] = fixed_slit
	                                _im.flush()
	
	                if extend_wavelengths:
	
	                    if by_source & (msametf is not None):
	                        # Run by individual source IDs
	                        rate_file = single_exposure_groups[mode][0]
	
	                        msa = msaexp.msa.MSAMetafile(msametf)
	                        msa.merge_source_ids()
	                        msa.write(prefix="", overwrite=True)
	
	                        source_ids = msaexp.msa.get_msa_source_ids(rate_file)
	
	                        with pyfits.open(rate_file, mode="update") as im:
	                            if "src" not in im[0].header["MSAMETFL"]:
	                                im[0].header["MSAMETFL"] = "src_" + msametf
	
	                            im.flush()
	
	                        # Run by source_id
	                        for source_id in source_ids:
	                            done_files = glob.glob(f"*_{source_id}.fits")
	                            if len(done_files) > 0:
	                                print(f"Skip completed {done_files[0]}")
	
	                            msametfl = msaexp.msa.pad_msa_metafile(
	                                msametf,
	                                pad=0,
	                                positive_ids=True,
	                                source_ids=[source_id],
	                                slitlet_ids=None,
	                                primary_sources=True,
	                            )
	
	                            try:
	                                pipe = pipeline_extended.run_pipeline(
	                                    rate_file,
	                                    slit_index=0,
	                                    all_slits=True,
	                                    write_output=True,
	                                    set_log=True,
	                                    skip_existing_log=False,
	                                    undo_flat=undo_flat,
	                                )
	                            except ValueError:
	                                msg = (
	                                    f"Failed to process source_id={source_id}"
	                                )
	                                utils.log_comment(
	                                    utils.LOGFILE, msg, verbose=True
	                                )
	                                continue
	
	                        # Set it back
	                        with pyfits.open(rate_file, mode="update") as im:
	                            if "src" not in im[0].header["MSAMETFL"]:
	                                im[0].header["MSAMETFL"] = msametf
	
	                            im.flush()
	
	                    else:
	                        pipe = pipeline_extended.run_pipeline(
	                            single_exposure_groups[mode][0],
	                            slit_index=0,
	                            all_slits=True,
	                            write_output=True,
	                            set_log=True,
	                            skip_existing_log=False,
	                            undo_flat=undo_flat,
	                        )
	
	                        if fixed_slit:
	                            photom_file = f"{use_prefix}_fs-photom.fits"
	                        else:
	                            photom_file = f"{use_prefix}_photom.fits"
	
	                        print(f"Write {photom_file}")
	                        pipe.write(photom_file)
	
	                else:
	                    try:
	                        pipe = pipeline.NirspecPipeline(
	                            mode=xmode,
	                            files=single_exposure_groups[mode],
	                            source_ids=source_ids,
	                            pad=pad,
	                            positive_ids=positive,  # Ignore background slits
	                        )
	
	                        pipe.full_pipeline(
	                            run_extractions=False,
	                            initialize_bkg=False,
	                            load_saved=None,
	                            scale_rnoise=False,
	                            fix_rows=False,
	                        )
	
	                    except NoDataOnDetectorError:
	                        print("NoDataOnDetectorError - skip")
	                        pipe = None
	
	                if fixed_slit:
	                    for _file in single_exposure_groups[mode]:
	                        with pyfits.open(_file, mode="update") as _im:
	                            if ORIG_EXPTYPE == "NRS_MSASPEC":
	                                print(
	                                    f"Reset {_file} FIXEDSLIT > MSA keywords"
	                                )
	                                _im[0].header["EXP_TYPE"] = "NRS_MSASPEC"
	                                _im[0].header["APERNAME"] = "NRS_FULL_MSA"
	                                _im[0].header["OPMODE"] = "MSASPEC"
	                                _im[0].header.pop("FXD_SLIT")
	                                _im.flush()
	
	                del pipe
	
	                os.remove(f"{xmode}.start")
	                print(f"{xmode} - Done! {time.ctime()}")
	
	            else:
	                print(f"Already completed: {mode}")
	
	            os.system(f"cat {mode}.log.txt >> {_NEW_LOGFILE}")

	
	    utils.LOGFILE = _NEW_LOGFILE
	
	    # # Sync slitlets to S3
	    # if outroot.split('-')[0] in ['macs0417','macs1423','macs0416','abell370']:
	    #     s3path = 'grizli-canucs/nirspec'
	    # else:
	    #     s3path = 'msaexp-nirspec/extractions'
	
	    # if (outroot not in ['uncover-deep-v1']) & (1):
	    #     msg = f'Sync slitlets to s3://{s3path}/slitlets/{outroot}/'
	    #     utils.log_comment(utils.LOGFILE, msg, verbose=True)
	
	    #     os.system(f'aws s3 sync ./ s3://{s3path}/slitlets/{outroot}/ --exclude "*" --include "*phot.*" --include "*raw.*" --include "*photom.*" --acl public-read --quiet')
	
	    if use_prefix != file_prefix:
	        _USE_LOGFILE = os.path.join(WORKPATH, use_prefix + "_rate.log.txt")
	        os.system(f"cp {_NEW_LOGFILE} {_USE_LOGFILE}")
	
	    # if os.path.exists(NIRSPEC_HOME):
	    #     local_path = os.path.join(NIRSPEC_HOME, outroot)
	    #     if not os.path.exists(local_path):
	    #         os.makedirs(local_path)
	
	    #     msg = f'cp {WORKPATH}/{use_prefix}* {local_path}/'
	    #     utils.log_comment(utils.LOGFILE, msg, verbose=True)
	    #     os.system(msg)
	
	    #     msg = f'sudo chown -R ec2-user {local_path}/'
	    #     utils.log_comment(utils.LOGFILE, msg, verbose=True)
	    #     os.system(msg)
	
	    #     if msametf is not None:
	    #         msg = f'cp {WORKPATH}/{msametf} {local_path}/'
	    #         utils.log_comment(utils.LOGFILE, msg, verbose=True)
	    #         os.system(msg)
	
	    utils.LOGFILE = _ORIG_LOGFILE
	
	    # if clean:
	    #     print('Clean up')
	    #     files = glob.glob('*')
	    #     for file in files:
	    #         print(f'rm {file}')
	    #         os.remove(file)
	
	    #     os.chdir(HOME)
	    #     os.rmdir(WORKPATH)
	
	    return 2	