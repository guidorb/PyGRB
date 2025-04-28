import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import c
from astropy.cosmology import WMAP9 as cosmo
from time import sleep
# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
from astropy.coordinates import SkyCoord, Distance
from uncertainties import ufloat
from uncertainties import unumpy as unp
from astropy.cosmology import Planck15
from astropy.io import ascii
from astropy.io import fits as pyfits
import pickle
from . import analysis_functions as af
from astropy.nddata import Cutout2D
from astropy.wcs import WCS


# def load_jewels(import_catalog=True, import_spectra=False):
# 	if import_catalog==True:
# 		tab_cat = ascii.read('/Users/guidorb/Dropbox/Catalogs/JEWELS/highz_msaid_public.dat')
# 	if import_spectra==True:
# 		tab_spec = pickle.load(open('/Users/guidorb/Dropbox/Catalogs/JEWELS/spectra_22April2024_single_public.p', "rb"), encoding='latin1')

# 	if (import_catalog==True) & (import_spectra==True):
# 		return tab_cat, tab_spec
# 	elif (import_catalog==True) & (import_spectra==False):
# 		return tab_cat
# 	elif (import_catalog==False) & (import_spectra==True):
# 		return tab_spec

class load_jewels:
	def __init__(self):
		self.tab = ascii.read('/Users/guidorb/Dropbox/Catalogs/JEWELS/highz_msaid_full.dat')
		self.spectra = pickle.load(open('/Users/guidorb/Dropbox/Catalogs/JEWELS/spectra_18April2025_full.p', "rb"), encoding='latin1')
		self.coords = SkyCoord(self.tab['ra'], self.tab['dec'], unit=(u.deg,u.deg))
		self.line_fluxes = ascii.read('/Users/guidorb/Dropbox/Catalogs/JEWELS/highz_msaid_full_linefluxes.cat')

		f = pyfits.open('/Users/guidorb/Dropbox/Catalogs/Astrodeep/AllFields_optap.fits')
		self.astrodeep = f[1].data.copy()
		self.astrodeep_coords = SkyCoord(f[1].data['RA'], f[1].data['DEC'], unit=(u.deg,u.deg))
		f.close()

		f = pyfits.open('/Users/guidorb/Dropbox/Catalogs/Astrodeep/AllFields_photoz.fits')
		self.astrodeep_zphot = f[1].data.copy()
		f.close()


	def plot_prism_spectrum(self, msaid, **kwargs):
		af.plot_prism_spectrum(self.spectra, msaid, **kwargs)

	def plot_grating_spectrum(self, msaid, **kwargs):
		af.plot_grating_spectrum(self.spectra, msaid, **kwargs)

	def load_prism_spectrum(self, msaid, units='msaexp'):
		if units=='msaexp':
			lam, flux, err = self.spectra[msaid]['prism-clear']['lam'], self.spectra[msaid]['prism-clear']['flux'], self.spectra[msaid]['prism-clear']['err']
		elif units in ['cgs','CGS','flam','FLAM']:
			lam, flux, err = self.spectra[msaid]['prism-clear']['lam'], self.spectra[msaid]['prism-clear']['flux'], self.spectra[msaid]['prism-clear']['err']
			lam = lam * 10000.
			flux = fnu_to_flam(lam, flux*(1.e-6)*(1.e-23), fnu_err=err*(1.e-6)*(1.e-23))
			err = unp.std_devs(flux)
			flux = unp.nominal_values(flux)
		return lam, flux, err

	def load_grating_spectrum(self, msaid, units='msaexp'):
		lam, flux, err = [], [], []
		for R in self.spectra[msaid].keys():
			if (R in ['prism-clear']) | ('h' in R):
				continue
			lam = np.concatenate([lam, self.spectra[msaid][R]['lam']])
			flux = np.concatenate([flux, self.spectra[msaid][R]['flux']])
			err = np.concatenate([flux, self.spectra[msaid][R]['err']])
		i = np.argsort(lam)
		lam = lam[i]
		flux = flux[i]
		err = err[i]

		if units=='msaexp':
			return lam, flux, err
		if units in ['cgs','CGS','flam']:
			lam = lam * 10000.
			flux = fnu_to_flam(lam, flux*(1.e-6)*(1.e-23), fnu_err=err*(1.e-6)*(1.e-23))
			err = unp.std_devs(flux)
			flux = unp.nominal_values(flux)
			return lam, flux, err

	def match_with_jewels(self, coord_str, unit='deg', sep=0.015):
		if unit=='deg':
			c = SkyCoord(coord_str, unit=(u.deg,u.deg))
		elif unit=='hour':
			c = SkyCoord(coord_str, unit=(u.hour,u.deg))
		
		seps = c.separation(self.coords).arcsec
		i = np.where(seps < sep)[0]
		if len(i)==0:
			print(f'No matches within {sep} arcsec')
		else:
			msaids = self.tab['msaid'][i]
			print(f"Closest match(es): {msaids.value} ({seps[i]} arcsec)")

	def write_spec_to_file(self, msaid):
		lam, flux, err = self.load_prism_spectrum(msaid, units='msaexp')

		f = open(f"{msaid}_nirspec_prism.txt", "w")
		f.write('# lam fnu fnu_err\n')
		for i in range(len(lam)):
			if (np.isfinite(flux[i])==False) | (np.isfinite(err[i])==False):
				continue
			else:
				f.write('%0.5f %0.5f %0.5f\n'%(lam[i],flux[i],err[i]))
		f.close()
		print(f'Written to file: ./{msaid}_nirspec_prism.txt')

	def get_entry(self, msaid, with_header=False):
		try:
			i = np.where(self.tab['msaid'] == msaid)[0][0]
			ra, dec, z, mu, AB = self.tab['ra'][i], self.tab['dec'][i], self.tab['z'][i], self.tab['mu'][i], self.tab['AB'][i]
			if with_header==True:
				print('MSA-ID Ra Dec zspec AB mu')
				print('--------------------------')
			print(msaid, '%0.5f %0.5f %0.3f %0.1f %0.1f'%(ra, dec, z, AB, mu))
		except:
			print('This MSA-ID entry does not exist.')




def separation(coord1, coord2, unit1, unit2):
	if unit1=='hour':
		c1 = SkyCoord(coord1, unit=(u.hour,u.deg))
	elif unit1=='deg':
		c1 = SkyCoord(coord1, unit=(u.deg,u.deg))

	if unit2=='hour':
		c2 = SkyCoord(coord2, unit=(u.hour,u.deg))
	elif unit2=='deg':
		c2 = SkyCoord(coord2, unit=(u.deg,u.deg))
	sep = c1.separation(c2)
	return sep
	
def calc_telescope_blind_offset(bright_object, faint_object):
	"""
	Calculate the offset in RA and DEC needed for e.g., a telescope blind offset 
	from a bright star to a faint galaxy. Takes into account the necessary 
	signs.

	bright_object : str
		Should be a string containing the RA and DEC coordinates in sexagesimal
		of the object to slew FROM.
		e.g., '09:59:55.9353 +01:58:49.1900'
	faint_object : str
		The same as bright_object, but needs to be the object that the telescope 
		will slew TO.

	"""
	bright_coord = SkyCoord(bright_object, unit=(u.hour,u.deg))
	faint_coord = SkyCoord(faint_object, unit=(u.hour,u.deg))

	dra, ddec = bright_coord.spherical_offsets_to(faint_coord)
	print(dra.to(u.arcsec), ddec.to(u.arcsec))

def SmallNStats(Ndet,Ntot):
	"""
	Ndet = int
		The number of detections of within a sample of Ntot.
	Ntot = int
		The total sample size within which Ndet is drawn.
	e.g., if one has 2 Lya detections in a sample of 8 galaxies, 
	then Ndet = 2 and Ntot = 8 and the resulting fraction should be
	f = 0.25^{+0.33}_{-0.16}
	"""
	G86_low = np.genfromtxt('/Users/guidorb/Dropbox/Postdoc/Gehrels_lowlim_conf.dat', usecols=(0,1), skip_header=1)
	G86_high = np.genfromtxt('/Users/guidorb/Dropbox/Postdoc/Gehrels_uplim_conf.dat', usecols=(0,1), skip_header=1)
	if Ndet > 0:
		Ndet_low = G86_low.T[1][(G86_low.T[0].astype(int) == Ndet)][0]
		Ndet_high = G86_high.T[1][(G86_high.T[0].astype(int) == Ndet)][0]
		frac = Ndet/Ntot
		frac_low = Ndet_low/Ntot
		frac_high = Ndet_high/Ntot
		sig_low = frac-frac_low
		sig_high = frac_high-frac
		if frac==1.:
			sig_high = 0.
		elif frac==0.:
			sig_low = 0.
		if frac + sig_high > 1.:
			sig_high = 1. - frac
		if frac - sig_low < 0.:
			sig_low = frac
	else:
		Ndet_high = G86_high.T[1][(G86_high.T[0].astype(int) == Ndet)][0]
		frac = 0.
		frac_high = Ndet_high/Ntot
		sig_low = 0.
		sig_high = frac_high - frac
	print('Fraction:',frac, 'Upper limit:',frac+sig_high, 'Lower limit:',frac-sig_low)
	print('Lower 1sig:', sig_low, 'Upper 1sig:', sig_high)
	# return frac, sig_high, sig_low, frac+sig_high, frac-sig_low

def M_from_m(mab, z=7., beta=-2):
	"""
	Compute absolute from apparent magnitude
	using K-correction
	"""
	# luminosity distance
	DL = Planck15.luminosity_distance(z)
	# K-correction
	K_corr = ((-beta - 1) * 2.5 * np.log10(1.0 + z))
	# Apparent mag
	MAB = mab - 5.0 * (np.log10(DL.value*1E6) - 1.0) + K_corr
	return MAB

def find_nearest(array,value,returnindex=True):
	idx = (np.abs(array-value)).argmin()
	if returnindex==True:
		return idx
	else:
		return array[idx]

def style_axes(ax, xlabel=None, ylabel=None, fontsize=None, labelsize=None, linewidth=None):
	if fontsize==None:
		fontsize = 18.5
	if labelsize==None:
		labelsize = 18.5
	if linewidth==None:
		linewidth = 1.25
	ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelsize=labelsize)
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(linewidth)
	ax.xaxis.set_tick_params(width=linewidth)
	ax.yaxis.set_tick_params(width=linewidth)
	if (xlabel != None) | (ylabel != None):
		ax.set_xlabel(xlabel, fontsize=fontsize)
		ax.set_ylabel(ylabel, fontsize=fontsize)

def flux_to_AB(flux, flux_err=None, unit='jy'):
	## NB: flux and flux_err have to be in units of 'jy' or 'fnu', specified by the input argument
	## a flux error must be provided, however one can set this to zero if need be.
	assert (unit=='fnu') | (unit=='jy'), "'units' variable must be 'fnu' or 'jy'!" 

	import uncertainties.unumpy as unp
	from uncertainties import ufloat

	if flux_err is not None:
		if np.size(flux) == 1:
			fluxpoint = ufloat(flux, flux_err)
		else:
			fluxpoint = unp.uarray(flux, flux_err)
	else:
		fluxpoint = flux

	if unit=='jy':
		if flux_err is not None:
			mag = 2.5 * (23. - unp.log10(fluxpoint)) - 48.6
		else:
			mag = 2.5 * (23. - np.log10(fluxpoint)) - 48.6
	elif unit=='fnu':
		if flux_err is not None:
			mag = 2.5 * (23. - unp.log10(fluxpoint / (1.e-23))) - 48.6
		else:
			mag = 2.5 * (23. - np.log10(fluxpoint / (1.e-23))) - 48.6
	return mag

def AB_to_flux(mag, mag_err=None, output_unit='jy'):
	## NB: mag and mag_err HAVE TO BE IN AB!!!
	## output can be either 'fnu' or 'jy' and must be a string!
	## a magnitude error must be provided, however one can set this to zero if need be.
	assert (output_unit=='fnu') | (output_unit=='jy'), "'output' variable must be 'fnu' or 'jy'!" 

	import uncertainties.unumpy as unp
	from uncertainties import ufloat

	if mag_err is not None:
		if np.size(mag) == 1:
			magpoint = ufloat(mag, mag_err)
		else:
			magpoint = unp.uarray([mag, mag_err])
	else:
		magpoint = mag
	# NB: magpoint is the apparent magnitude and must be in AB magnitudes!
	jy = 10**(23.-(magpoint+48.6)/2.5)
	fnu = jy*(10**-23)
	if output_unit=='fnu':
		return fnu
	elif output_unit=='jy':
		return jy

def calc_IRAC_color(mAB, EW, z, lam_eff, filter='CH2'):
	assert mAB!=None, 'mAB needs to be the apparent (AB) magnitude of the continuum!'
	assert EW!=None, 'EW needs to be the (rest-frame) EW of the line(s) in the relevant filter'
	assert z!=None, 'A redshift, z, must be provided!'
	assert lam_eff!=None, 'An effective wavelength (in Angstroms) of "filter" must be provided!'
	assert filter!=None, 'filter cannot be None!'
	
	from uncertainties import unumpy as unp
	
	if filter=='CH1':
		filt_lam, filt_response = np.genfromtxt('/Users/guidorb/Dropbox/Postdoc/filters/Spitzer_IRAC.CH1', unpack=True)
	elif filter=='CH2':
		filt_lam, filt_response = np.genfromtxt('/Users/guidorb/Dropbox/Postdoc/filters/Spitzer_IRAC.CH2', unpack=True)
	
	flux_fnu = unp.nominal_values(convert_AB_to_flux(mAB, 0., output='fnu'))  # convert the continuum magnitude to a flux density
	flux_flam = fnu_to_flam(flux_fnu, lam_eff)  # convert between flux density for the right units: fnu to flam
	
	EW_z = EW * (1.+z)  # get the redshifted EW of the line(s)
	lineflux = EW_z * flux_flam  # convert this to an integrated lineflux by multiplying by the continuum value (see definition of EW)

	idx = np.where(filt_response > 0.5)  # essentially the FWHM of the filter
	filter_width = max(filt_lam[idx]) - min(filt_lam[idx])  # this needs to remain in Angstroms!
	
	lineflux_filter = lineflux / filter_width  # spread the integrated flux of the line over the effective width of the filter in question
	
	color = -2.5 * np.log10(flux_flam/lineflux_filter)
	return color

def calc_IRAC_EW(mAB_cont, color, z):
	assert mAB_cont!=None, 'mAB needs to be the apparent (AB) magnitude of the continuum (e.g., 3.6 micron magnitude!'
	assert color!=None, 'color needs to be the [3.6]-[4.5] color in magnitudes'
	assert z!=None, 'A redshift, z, must be provided!'
	
	from uncertainties import unumpy as unp    

	filt_lam, filt_response = np.genfromtxt('/Users/guidorb/Dropbox/Postdoc/filters/Spitzer_IRAC.CH2', unpack=True)
	
	line_mag = mAB_cont - color
	ratio = (10**((mAB_cont-line_mag)/-2.5)) # ratio in flux units, to be used lower down
	
	flux_fnu_cont = unp.nominal_values(convert_AB_to_flux(mAB_cont, 0., output_unit='fnu'))
	flux_flam_cont = fnu_to_flam(flux_fnu_cont, 45000.) # units of erg / s cm2 A
	lineflux_filter = flux_flam_cont/ratio # line flux in units of erg / s cm2 A

	idx = np.where(filt_response > 0.5)  # essentially the FWHM of the filter
	filter_width = max(filt_lam[idx]) - min(filt_lam[idx])  # this needs to remain in Angstroms!
	lineflux = lineflux_filter * filter_width # in units of erg / s cm2
	
	EW = lineflux/flux_flam_cont   # the observed EW (in Angstroms). Basically erg / s cm2 divided by erg / s cm2 A
	EW0 = EW / (1.+z)   # the rest-frame EW (in Angstroms)
	return EW0, EW

def calc_sky_sep(coord1, coord2, return_units='degree'):
	coord_a = SkyCoord(ra=coord1[0], dec=coord1[1])
	coord_b = SkyCoord(ra=coord2[0], dec=coord2[1])
	seps = coord_a.separation(coord_b)
	
	if return_units=='arcsec':
		return seps.to(u.arcsec)
	elif return_units=='arcmin':
		return seps.to(u.arcmin)
	elif return_units=='degree':
		return seps
	
def fnu_to_flam(lam, fnu, fnu_err=None):
	'''lam : the wavelength at which to evaluate flambda, in units of angstroms
		fnu : the flux density in units of erg/s/cm^2/Hz (NB: NOT JANSKIES!)
	'''
	from astropy.constants import c
	import astropy.units as u
	
	if fnu_err is not None:
		spec = unp.uarray(fnu,fnu_err)
	else:
		spec = fnu#.copy()

	c = c.to(u.AA/u.s).value
	flam = (c/(lam**2.)) * spec# * (u.erg/u.s/(u.cm**2.)/u.AA)
	return flam

def flam_to_fnu(lam, flam, flam_err=None):
	'''flam : the flux density in units of erg/s/cm^2/A
	   lam : the wavelength at which to evaluate flambda, in units of angstroms'''
	from astropy.constants import c
	import astropy.units as u
	
	if flam_err is not None:
		spec = unp.uarray(flam,flam_err)
	else:
		spec = flam#.copy()

	c = c.to(u.AA/u.s).value
	fnu = ((lam**2.)/c) * spec# * (u.erg/u.s/(u.cm**2.)/u.Hz)
	return fnu

def emission_line(xvals, x0, A, width, FWHM=False):
	if FWHM==False:
		linewidth = width
	else:
		linewidth = width / 2. * np.sqrt(2. * np.log(2.))
	# gauss = A * np.exp(-((x0-xvals)**2.)/(2.*linewidth**2.))
	gauss = A * np.exp(-np.power(xvals - x0, 2.) / (2. * np.power(linewidth, 2.)))
	return gauss

def photo_from_filter(wave, spectrum, filt=None):
	# NB: wave should be in microns
	path = '/Users/guidorb/Dropbox/Postdoc/filters/'

	filt_file = np.genfromtxt(path+filt, unpack=False)
	response = np.interp(wave, filt_file.T[0]/10000, filt_file.T[1], left=0., right=0.)

	photo = np.nansum(spectrum * response)/np.nansum(response)
	return  photo

def get_filter_info(filt, output_unit='mu'):
	f = ascii.read(f'/Users/guidorb/Dropbox/Postdoc/filters/{filt}')
	f['col2'] = f['col2'] / np.max(f['col2'])
	if output_unit=='mu':
		f['col1'] = f['col1'] / 10000.
	
	idx = np.where(f['col2'] > 0.5)[0]
	lam_cent = np.median(f['col1'][idx])
	lam_min, lam_max = np.min(f['col1'][idx]), np.max(f['col1'][idx])
	lam_err_low = lam_cent - lam_min
	lam_err_high = lam_max - lam_cent
	
	return lam_cent, [[lam_err_low],[lam_err_high]], [lam_min,lam_max]

# def photo_from_filter(wave, spectrum, spectrum_err=None, mask=None, filt=None, threshold=90):
# 	# NB: wave should be in microns
# 	path = '/Users/guidorb/Dropbox/Postdoc/filters/'
# 	f = ascii.read(path+filt)
	
# 	filt_lams, filt_resp = f['col1'], f['col2']
# 	filt_lams = filt_lams / 10000.
# 	filt_resp = filt_resp / np.nanmax(filt_resp)
# 	lam_cent = np.mean(filt_lams[(filt_resp > 0.5)])
# 	lam_err = np.mean([np.max(filt_lams[(filt_resp > 0.5)])-lam_cent, lam_cent - np.min(filt_lams[(filt_resp > 0.5)])])
	
# 	response = np.interp(wave, f['col1']/10000, f['col2'], left=0., right=0.)
# 	norm_response = response / np.max(response)
# 	idx = np.where(norm_response > 0.5)[0]
# 	N = len(idx)
# 	Ngood = len(np.where(spectrum[idx] != 0.)[0])
# 	if Ngood < int((threshold/100.) * N):
# 		print(f'Number of non-zero flux pixels is below required threshold ({threshold}\%)')
# 		return lam_cent, lam_err, 0., 0.

# 	if mask is None:
# 		mask = np.zeros_like(wave)
	
# 	if spectrum_err is not None:
# 		spec = unp.uarray([spectrum, spectrum_err])
# 		photo = np.nansum(spec[(mask==0)] * response[(mask==0)])/np.nansum(response[(mask==0)])
# 		photo_err = unp.std_devs(photo)
# 		photo = unp.nominal_values(photo)
# 		return lam_cent, lam_err, photo, photo_err
# 	else:
# 		spec = spectrum.copy()
# 		photo = np.nansum(spec[(mask==0)] * response[(mask==0)])/np.nansum(response[(mask==0)])
# 		return lam_cent, lam_err, photo


def lighten_color(color, amount=1.):
	"""
	Lightens the given color by multiplying (1-luminosity) by the given amount.
	Input can be matplotlib color string, hex string, or RGB tuple.

	Examples:
	>> lighten_color('g', 0.3)
	>> lighten_color('#F034A3', 0.6)
	>> lighten_color((.3,.55,.1), 0.5)
	"""
	import matplotlib.colors as mc
	import colorsys
	try:
		c = mc.cnames[color]
	except:
		c = color
	c = colorsys.rgb_to_hls(*mc.to_rgb(c))
	rgbval = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
	return rgbval

def get_continuous_cmap(hex_list, float_list=None):
	import matplotlib.colors as mcolors
	
	def hex_to_rgb(value):
		'''
		Converts hex to rgb colours
		value: string of 6 characters representing a hex colour.
		Returns: list length 3 of RGB values'''
		value = value.strip("#") # removes hash symbol if present
		lv = len(value)
		return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
	
	def rgb_to_dec(value):
		'''
		Converts rgb to decimal colours (i.e. divides each value by 256)
		value: list (length 3) of RGB values
		Returns: list (length 3) of decimal values'''
		return [v/256 for v in value]
	
	''' creates and returns a color map that can be used in heat map figures.
		If float_list is not provided, colour map graduates linearly between each color in hex_list.
		If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
		
		Parameters
		----------
		hex_list: list of hex code strings
		float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
		
		Returns
		----------
		colour map'''
	rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
	if float_list:
		pass
	else:
		float_list = list(np.linspace(0,1,len(rgb_list)))
		
	cdict = dict()
	for num, col in enumerate(['red', 'green', 'blue']):
		col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
		cdict[col] = col_list
	cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
	return cmp

def convolve_spectrum(lam, flux, filter_name, filter_dir=None):
	"""
	Convolve an input spectrum to an instrument filter response curve to get photometry

	Inputs:
		lam [microns] : the wavelength array of the spectrum
		flux [any unit] : the flux of the spectrum
		filter_name [str] : the name of the filter response curve file. Assumes a two-column file, where the first column is the wavelength in angstroms and the second column is the response curve. Also assumes no commented lines
		filter_dir [str] : the directory containing all the filter response curve. If false, assumes a default directory
	Returns:
		The central wavelength of the filter response curve and the convolved flux of the spectrum in the units that it was input.
	"""
	if filter_dir==None:
		filter_dir = '/Users/guidorb/Dropbox/Postdoc/filters/'

	filt_lam, filt_resp = np.genfromtxt(filter_dir + filter_name, unpack=True)
	filt_resp = np.interp(lam, filt_lam/10000., filt_resp)

	flux_val = np.nansum(filt_resp * flux)/np.nansum(filt_resp)
	lam_cent = np.median(lam[((filt_resp/np.nanmax(filt_resp)) > 0.5)])
	return lam_cent, flux_val

def print_eazy_filter(N, filter_info_path=None):
	if filter_info_path==None:
		filter_info_path = '/Users/guidorb/AstroSoftware/eazy/inputs/'
	f = open(filter_info_path + 'FILTER.RES.latest.info', "r")

	dictionary = {}
	for line in f.readlines():
		if int(line.split(' ')[0])==N:
			print(line)


def get_Inoue14_trans(rest_wavs, z_obs):
	""" Calculate IGM transmission using Inoue et al. (2014) model. """
	coefs = np.loadtxt("/Users/guidorb/Dropbox/Postdoc/lyman_series_coefs_inoue_2014_table2.txt")

	if isinstance(rest_wavs, float):
		rest_wavs = np.array([rest_wavs])

	tau_LAF_LS = np.zeros((39, rest_wavs.shape[0]))
	tau_DLA_LS = np.zeros((39, rest_wavs.shape[0]))
	tau_LAF_LC = np.zeros(rest_wavs.shape[0])
	tau_DLA_LC = np.zeros(rest_wavs.shape[0])

	# Populate tau_LAF_LS
	for j in range(39):

		if z_obs < 1.2:
			wav_slice = ((rest_wavs*(1.+z_obs) > coefs[j, 1])
						 & (rest_wavs*(1.+z_obs)
							< (1+z_obs)*coefs[j, 1]))

			tau_LAF_LS[j, wav_slice] = (coefs[j, 2]
										* (rest_wavs[wav_slice]
										   * (1.+z_obs)/coefs[j, 1])**1.2)

		elif z_obs < 4.7:
			wav_slice_1 = ((rest_wavs*(1.+z_obs) > coefs[j, 1])
						   & (rest_wavs*(1.+z_obs) < 2.2*coefs[j, 1]))
			wav_slice_2 = ((rest_wavs*(1.+z_obs) > 2.2*coefs[j, 1])
						   & (rest_wavs*(1.+z_obs)
							  < (1+z_obs)*coefs[j, 1]))

			tau_LAF_LS[j, wav_slice_1] = (coefs[j, 2]
										  * (rest_wavs[wav_slice_1]
											 * (1.+z_obs)/coefs[j, 1])**1.2)

			tau_LAF_LS[j, wav_slice_2] = (coefs[j, 3]
										  * (rest_wavs[wav_slice_2]
											 * (1.+z_obs)/coefs[j, 1])**3.7)

		else:
			wav_slice_1 = ((rest_wavs*(1.+z_obs) > coefs[j, 1])
						   & (rest_wavs*(1.+z_obs) < 2.2*coefs[j, 1]))

			wav_slice_2 = ((rest_wavs*(1.+z_obs) > 2.2*coefs[j, 1])
						   & (rest_wavs*(1.+z_obs) < 5.7*coefs[j, 1]))

			wav_slice_3 = ((rest_wavs*(1.+z_obs) > 5.7*coefs[j, 1])
						   & (rest_wavs*(1.+z_obs)
							  < (1+z_obs)*coefs[j, 1]))

			tau_LAF_LS[j, wav_slice_1] = (coefs[j, 2]
										  * (rest_wavs[wav_slice_1]
											 * (1.+z_obs)/coefs[j, 1])**1.2)

			tau_LAF_LS[j, wav_slice_2] = (coefs[j, 3]
										  * (rest_wavs[wav_slice_2]
											 * (1.+z_obs)/coefs[j, 1])**3.7)

			tau_LAF_LS[j, wav_slice_3] = (coefs[j, 4]
										  * (rest_wavs[wav_slice_3]
											 * (1.+z_obs)/coefs[j, 1])**5.5)

	# Populate tau_DLA_LS
	for j in range(39):

		if z_obs < 2.0:
			wav_slice = ((rest_wavs*(1.+z_obs) > coefs[j, 1])
						 & (rest_wavs*(1.+z_obs)
							< (1+z_obs)*coefs[j, 1]))

			tau_DLA_LS[j, wav_slice] = (coefs[j, 5]
										* (rest_wavs[wav_slice]
										   * (1.+z_obs)/coefs[j, 1])**2.0)

		else:
			wav_slice_1 = ((rest_wavs*(1.+z_obs) > coefs[j, 1])
						   & (rest_wavs*(1.+z_obs) < 3.0*coefs[j, 1]))

			wav_slice_2 = ((rest_wavs*(1.+z_obs) > 3.0*coefs[j, 1])
						   & (rest_wavs*(1.+z_obs) < (1+z_obs)
							  * coefs[j, 1]))

			tau_DLA_LS[j, wav_slice_1] = (coefs[j, 5]
										  * (rest_wavs[wav_slice_1]
											 * (1.+z_obs)/coefs[j, 1])**2.0)

			tau_DLA_LS[j, wav_slice_2] = (coefs[j, 6]
										  * (rest_wavs[wav_slice_2]
											 * (1.+z_obs)/coefs[j, 1])**3.0)

	# Populate tau_LAF_LC
	if z_obs < 1.2:
		wav_slice = ((rest_wavs*(1.+z_obs) > 911.8)
					 & (rest_wavs*(1.+z_obs) < 911.8*(1.+z_obs)))

		tau_LAF_LC[wav_slice] = (0.325*((rest_wavs[wav_slice]
										 * (1.+z_obs)/911.8)**1.2
										- (((1+z_obs)**-0.9)
										   * (rest_wavs[wav_slice]
										   * (1.+z_obs)/911.8)**2.1)))

	elif z_obs < 4.7:
		wav_slice_1 = ((rest_wavs*(1.+z_obs) > 911.8)
					   & (rest_wavs*(1.+z_obs) < 911.8*2.2))

		wav_slice_2 = ((rest_wavs*(1.+z_obs) > 911.8*2.2)
					   & (rest_wavs*(1.+z_obs) < 911.8*(1.+z_obs)))

		tau_LAF_LC[wav_slice_1] = (((2.55*10**-2)*((1+z_obs)**1.6)
									* (rest_wavs[wav_slice_1]
									* (1.+z_obs)/911.8)**2.1)
								   + (0.325*((rest_wavs[wav_slice_1]
									  * (1.+z_obs)/911.8)**1.2))
								   - (0.25*((rest_wavs[wav_slice_1]
											 * (1.+z_obs)/911.8)**2.1)))

		tau_LAF_LC[wav_slice_2] = ((2.55*10**-2)
								   * (((1.+z_obs)**1.6)
									  * ((rest_wavs[wav_slice_2]
										  * (1.+z_obs)/911.8)**2.1)
									  - ((rest_wavs[wav_slice_2]
										  * (1.+z_obs)/911.8)**3.7)))

	else:
		wav_slice_1 = ((rest_wavs*(1.+z_obs) > 911.8)
					   & (rest_wavs*(1.+z_obs) < 911.8*2.2))

		wav_slice_2 = ((rest_wavs*(1.+z_obs) > 911.8*2.2)
					   & (rest_wavs*(1.+z_obs) < 911.8*5.7))

		wav_slice_3 = ((rest_wavs*(1.+z_obs) > 911.8*5.7)
					   & (rest_wavs*(1.+z_obs) < 911.8*(1.+z_obs)))

		tau_LAF_LC[wav_slice_1] = (((5.22*10**-4)*((1+z_obs)**3.4)
									* (rest_wavs[wav_slice_1]
									   * (1.+z_obs)/911.8)**2.1)
								   + (0.325*(rest_wavs[wav_slice_1]
									  * (1.+z_obs)/911.8)**1.2)
								   - ((3.14*10**-2)*((rest_wavs[wav_slice_1]
									  * (1.+z_obs)/911.8)**2.1)))

		tau_LAF_LC[wav_slice_2] = (((5.22*10**-4)*((1+z_obs)**3.4)
									* (rest_wavs[wav_slice_2]
									   * (1.+z_obs)/911.8)**2.1)
								   + (0.218*((rest_wavs[wav_slice_2]
											 * (1.+z_obs)/911.8)**2.1))
								   - ((2.55*10**-2)*((rest_wavs[wav_slice_2]
													  * (1.+z_obs)
													  / 911.8)**3.7)))

		tau_LAF_LC[wav_slice_3] = ((5.22*10**-4)
								   * (((1+z_obs)**3.4)
									  * (rest_wavs[wav_slice_3]
										 * (1.+z_obs)/911.8)**2.1
									  - (rest_wavs[wav_slice_3]
										 * (1.+z_obs)/911.8)**5.5))

	# Populate tau_DLA_LC
	if z_obs < 2.0:
		wav_slice = ((rest_wavs*(1.+z_obs) > 911.8)
					 & (rest_wavs*(1.+z_obs) < 911.8*(1.+z_obs)))

		tau_DLA_LC[wav_slice] = (0.211*((1+z_obs)**2.)
								 - (7.66*10**-2)*(((1+z_obs)**2.3)
												  * (rest_wavs[wav_slice]
													 * (1.+z_obs)/911.8)**-0.3)
								 - 0.135*((rest_wavs[wav_slice]
										   * (1.+z_obs)/911.8)**2.0))

	else:
		wav_slice_1 = ((rest_wavs*(1.+z_obs) > 911.8)
					   & (rest_wavs*(1.+z_obs) < 911.8*3.0))

		wav_slice_2 = ((rest_wavs*(1.+z_obs) > 911.8*3.0)
					   & (rest_wavs*(1.+z_obs) < 911.8*(1.+z_obs)))

		tau_DLA_LC[wav_slice_1] = (0.634 + (4.7*10**-2)*(1.+z_obs)**3.
								   - ((1.78*10**-2)*((1.+z_obs)**3.3)
									  * (rest_wavs[wav_slice_1]
										 * (1.+z_obs)/911.8)**-0.3)
								   - (0.135*(rest_wavs[wav_slice_1]
											 * (1.+z_obs)/911.8)**2.0)
								   - 0.291*(rest_wavs[wav_slice_1]
											* (1.+z_obs)/911.8)**-0.3)

		tau_DLA_LC[wav_slice_2] = ((4.7*10**-2)*(1.+z_obs)**3.
								   - ((1.78*10**-2)*((1.+z_obs)**3.3)
									  * (rest_wavs[wav_slice_2]
										 * (1.+z_obs)/911.8)**-0.3)
								   - ((2.92*10**-2)
									  * (rest_wavs[wav_slice_2]
										 * (1.+z_obs)/911.8)**3.0))

	tau_LAF_LS_sum = np.sum(tau_LAF_LS, axis=0)
	tau_DLA_LS_sum = np.sum(tau_DLA_LS, axis=0)

	tau = tau_LAF_LS_sum + tau_DLA_LS_sum + tau_LAF_LC + tau_DLA_LC

	return np.exp(-tau)


def get_IGM_absorption(z_obs):
	rest_wavs = np.arange(0.5, 3000., 0.5)
	trans = get_Inoue14_trans(rest_wavs, z_obs)
	return rest_wavs*(1+z_obs), trans



def plot_filters(filt='all', z=7.):
	from astropy.io import ascii
	import numpy as np
	import matplotlib.pyplot as plt
	
	f = ascii.read('/Users/guidorb/Dropbox/papers/z5_Templates/gsf/input_stacks/All/z5p937_AllObjects_full.txt')
	ilam, iflux = (f['lambda(AA)']/10000.)/(1.+5.937), f['flux_nu']
	ilam = ilam * (1.+z)
	
	fig = plt.figure(figsize=(10.,6.))
	ax = fig.add_subplot(111)
	ax.step(ilam, iflux, color='darkgrey', linewidth=1.25)
	ax.set_xlim(0.5,6.)
	ax.set_xticks(np.arange(1.,7.,1))
	ax.set_ylim(-1.,3.5)
	
	
	if filt in ['NIRCam','all']:
		nircam_wide = ['JWST_NIRCam.F090W',
					   'JWST_NIRCam.F115W',
					   'JWST_NIRCam.F150W',
					   'JWST_NIRCam.F200W',
					   'JWST_NIRCam.F277W',
					   'JWST_NIRCam.F356W',
					   'JWST_NIRCam.F444W']
		for ext in nircam_wide:
			f = ascii.read(f'/Users/guidorb/Dropbox/Postdoc/filters/{ext}')
			lam_cent, _, _ = get_filter_info(ext, output_unit='mu')
			if (lam_cent < ax.set_xlim()[0]) | (lam_cent > ax.set_xlim()[1]):
				continue
			f['col1'] = f['col1'] / 10000.
			f['col2'] = ((f['col2'] / np.nanmax(f['col2'])) * (ax.set_ylim()[0] + ((ax.set_ylim()[1] - ax.set_ylim()[0]) * 0.3))) + ax.set_ylim()[0]
			ax.plot(f['col1'], f['col2'], color='darkred', linewidth=1.25)
			filtname = ext.split('.')[-1]
			ax.text(np.mean(f['col1']), np.max(f['col2']+0.2), s=filtname, color='darkred', fontsize=10, ha='center')

		ax.plot(f['col1'], f['col2'], color='darkred', linewidth=1.25, label='JWST/NIRCam (wide)')
		

		nircam_medium = ['JWST_NIRCam.F480M',
						 'JWST_NIRCam.F460M',
						 'JWST_NIRCam.F430M',
						 'JWST_NIRCam.F410M',
						 'JWST_NIRCam.F360M',
						 'JWST_NIRCam.F335M',
						 'JWST_NIRCam.F300M',
						 'JWST_NIRCam.F250M',
						 'JWST_NIRCam.F210M',
						 'JWST_NIRCam.F182M',
						 'JWST_NIRCam.F162M',
						 'JWST_NIRCam.F140M']
		for ext in nircam_medium:
			f = ascii.read(f'/Users/guidorb/Dropbox/Postdoc/filters/{ext}')
			lam_cent, _, _ = get_filter_info(ext, output_unit='mu')
			if (lam_cent < ax.set_xlim()[0]) | (lam_cent > ax.set_xlim()[1]):
				continue
			f['col1'] = f['col1'] / 10000.
			f['col2'] = ((f['col2'] / np.nanmax(f['col2'])) * (ax.set_ylim()[0] + ((ax.set_ylim()[1] - ax.set_ylim()[0]) * 0.4))) + ax.set_ylim()[0]
			ax.plot(f['col1'], f['col2'], color='darkorange', linewidth=1.25)
			filtname = ext.split('.')[-1]
			ax.text(np.mean(f['col1']), np.max(f['col2']+0.2), s=filtname, color='darkorange', fontsize=10, ha='center')
		ax.plot(f['col1'], f['col2'], color='darkorange', linewidth=1.25, label='JWST/NIRCam (medium)')
		
	if filt in ['HST','all']:
		hst_wfc3 = ['HST_WFC3.F098M',
					'HST_WFC3.F105W',
					'HST_WFC3.F125W',
					'HST_WFC3.F140W',
					'HST_WFC3.F160W']
		for ext in hst_wfc3:
			f = ascii.read(f'/Users/guidorb/Dropbox/Postdoc/filters/{ext}')
			lam_cent, _, _ = get_filter_info(ext, output_unit='mu')
			if (lam_cent < ax.set_xlim()[0]) | (lam_cent > ax.set_xlim()[1]):
				continue
			f['col1'] = f['col1'] / 10000.
			f['col2'] = ((f['col2'] / np.nanmax(f['col2'])) * (ax.set_ylim()[0] + ((ax.set_ylim()[1] - ax.set_ylim()[0]) * 0.25))) + ax.set_ylim()[0]
			ax.plot(f['col1'], f['col2'], color='navy', linewidth=1.25)
			filtname = ext.split('.')[-1]
			ax.text(np.mean(f['col1']), np.max(f['col2']+0.2), s=filtname, color='navy', fontsize=10, ha='center')
		ax.plot(f['col1'], f['col2'], color='navy', linewidth=1.25, label='HST/WFC3')
			
		hst_acs = ['HST_ACS.F814W',
				   'HST_ACS.F606W',
				   'HST_ACS.F435W']
		for ext in hst_acs:
			f = ascii.read(f'/Users/guidorb/Dropbox/Postdoc/filters/{ext}')
			lam_cent, _, _ = get_filter_info(ext, output_unit='mu')
			if (lam_cent < ax.set_xlim()[0]) | (lam_cent > ax.set_xlim()[1]):
				continue
			f['col1'] = f['col1'] / 10000.
			f['col2'] = ((f['col2'] / np.nanmax(f['col2'])) * (ax.set_ylim()[0] + ((ax.set_ylim()[1] - ax.set_ylim()[0]) * 0.25))) + ax.set_ylim()[0]
			ax.plot(f['col1'], f['col2'], color='royalblue', linewidth=1.25)
			filtname = ext.split('.')[-1]
			ax.text(np.mean(f['col1']), np.max(f['col2']+0.2), s=filtname, color='royalblue', fontsize=10, ha='center')
		ax.plot(f['col1'], f['col2'], color='royalblue', linewidth=1.25, label='HST/ACS')


	if filt in ['NIRSpec','all']:
		jwst_nirspec = ['JWST_NIRSpec.F140X',
						'JWST_NIRSpec.CLEAR',
						'JWST_NIRSpec.F110W']
		for ext in jwst_nirspec:
			f = ascii.read(f'/Users/guidorb/Dropbox/Postdoc/filters/{ext}')
			lam_cent, _, _ = get_filter_info(ext, output_unit='mu')
			if (lam_cent < ax.set_xlim()[0]) | (lam_cent > ax.set_xlim()[1]):
				continue
			f['col1'] = f['col1'] / 10000.
			f['col2'] = ((f['col2'] / np.nanmax(f['col2'])) * (ax.set_ylim()[0] + ((ax.set_ylim()[1] - ax.set_ylim()[0]) * 0.475))) + ax.set_ylim()[0]
			ax.plot(f['col1'], f['col2'], color='lightseagreen', linewidth=1.25)
			filtname = ext.split('.')[-1]
			ax.text(np.mean(f['col1']), np.max(f['col2']+0.2), s=filtname, color='lightseagreen', fontsize=10, ha='center')
		ax.plot(f['col1'], f['col2'], color='lightseagreen', linewidth=1.25, label='JWST/NIRSpec')
			
	
		ax.plot([0.7,1.27], [1.,1.], color='steelblue', linewidth=1.5, linestyle='--')
		ax.plot([0.97,1.89], [1.2,1.2], color='steelblue', linewidth=1.5, linestyle='--')
		ax.plot([1.66,3.17], [1.3,1.3], color='steelblue', linewidth=1.5, linestyle='--')
		ax.plot([2.87,5.27], [1.4,1.4], color='steelblue', linewidth=1.5, linestyle='--')
		ax.plot([0.6,5.3], [1.8,1.8], color='lightseagreen', linewidth=1.5, linestyle='--')
		
		ax.text(np.mean(ax.set_xlim()), 1.45, s='NIRSpec grating', color='steelblue', va='bottom', ha='center', fontsize=10)
		ax.text(np.mean(ax.set_xlim()), 1.85, s='NIRSpec prism', color='lightseagreen', va='bottom', ha='center', fontsize=10)
	ax.annotate(xy=(0.5,0.965), xycoords=('axes fraction'), text='z=%0.2f'%z, color='darkgrey', fontsize=20, ha='center', va='top')
		
	ax.legend(loc='upper left')
		
	style_axes(ax, r'Observed Wavelength [$\mu$m]', r'$F_{\nu}$ [arbitrary]')
	
	
	axx = ax.twiny()    
	xticks = []
	xtick_labels = []
	for x in np.arange(1000., 6000., 1000.):
		xz = (x * (1.+z)) / 10000.
		xticks.append(xz)
		xtick_labels.append(str(int(x)))
	axx.set_xticks(xticks)
	axx.set_xlim(ax.set_xlim())
	axx.set_xticklabels(xtick_labels)
	style_axes(axx, 'Rest Wavelength [$\AA$]')
	axx.tick_params(axis='x', top=True, bottom=False, which='both')
	axx.xaxis.labelpad = 12.5

	plt.tight_layout()
	plt.show()


def make_cutout(input_fits, output_fits, ra, dec, size_arcsec, plot_cutout=False, clim=None):
	# Open the FITS file
	with pyfits.open(input_fits) as hdul:
		data = hdul[0].data
		header = hdul[0].header
		wcs = WCS(header)

		# Convert size from arcsec to degrees
		size = (size_arcsec * u.arcsec).to(u.deg)

		# Define the center using SkyCoord
		center = SkyCoord(ra=ra, dec=dec, unit=(u.deg,u.deg), frame='fk5')

		# Create the cutout
		cutout = Cutout2D(data, position=center, size=size, wcs=wcs, copy=True)

		# Update the header with the cutout WCS
		new_header = cutout.wcs.to_header()
		for key in new_header:
			header[key] = new_header[key]

		# Replace data with the cutout and write new FITS
		hdul[0].data = cutout.data
		hdul[0].header = header
		hdul.writeto(output_fits, overwrite=True)

		print(f"Cutout saved to: {output_fits}")

		if plot_cutout==True:
			img = pyfits.getdata(output_fits)
			plt.figure(figsize=(5, 5))
			if clim==None:
				plt.imshow(img, origin='lower')
			else:
				plt.imshow(img, origin='lower', clim=(clim[0],clim[1]))
			plt.axis('off')
			plt.title('RGB Composite from HST Images')
			plt.show()