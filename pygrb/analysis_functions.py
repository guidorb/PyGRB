import glob
import numpy as np
# from jwst.datamodels import ImageModel, MultiSpecModel
from astropy.modeling import models, fitting
from astropy.visualization import astropy_mpl_style, simple_norm
# from specutils import Spectrum1D
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# from ipywidgets import interact
# import ipywidgets as widgets
from astropy.stats import sigma_clip
from copy import copy
import astropy.units as u
import pickle
from . import general_functions as gf
from astropy.io import ascii
from matplotlib import gridspec
from uncertainties import ufloat
from uncertainties import unumpy as unp
from astropy.constants import c
import os
from astropy.io import fits as pyfits
import random
from astropy.cosmology import Planck15 as cosmo

_C_KM_S = c.to(u.km/u.s).value

# Canonical emission line dictionary shared across plotting and masking functions.
# Keys are display labels; values are lists of rest-frame wavelengths in Angstroms.
_EMISSION_LINES = {
    r'Ly$\alpha$':           [1215.67],
    'C IV':                  [np.mean([1548.187, 1550.772])],
    r'He II + O III]':       [1640.42, np.mean([1660.809, 1666.150])],
    r'C III]':               [1908.734],
    '[O II]':                [3728.4835],
    '[Ne III]':              [3968.59, 3869.86],
    r'H$\delta$':            [4102.8922],
    r'H$\gamma$ + [O III]':  [4353.05985],
    r'He I_1':               [4472.734],
    r'He II':                [4687.015],
    r'H$\beta$':             [4862.6830],
    '[O III]':               [4960.295, 5008.240],
    'He I_2':                [5877.252],
    '[O I]':                 [6302.046],
    r'H$\alpha$':            [6564.608],
    '[S II]':                [6725.4845],
    'He I_3':                [7065.196],
}

# Grating-specific lines (adds NII doublet around Halpha)
_EMISSION_LINES_GRATING = {**_EMISSION_LINES, 'NII_1': [6549.86], 'NII_2': [6585.27]}

# Lines used for continuum masking
_MASK_LINES = {
    r'Ly$\alpha$': [1215.67],
    'OII':         [3727.092, 3729.875],
    'NeIII':       [3968.59, 3869.86],
    r'H$\delta$':  [4102.8922],
    'OIII':        [4364.436],
    r'H$\beta$':   [4862.6830],
    '[OIII]':      [4960.295, 5008.240],
    'HeI':         [5877.252],
    r'H$\alpha$':  [6564.608],
    'HeI_2':       [7065.196],
}

def read_composite(name, units='fnu'):
	corename = name.split('/')[-1]
	f1 = ascii.read(f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/input_stacks/{name}.txt')
	lam, flux, err = f1['lambda(AA)'], f1['flux_nu'], f1['eflux_nu_lines']
	
	f2 = pyfits.open(f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/output_stacks/gsf_spec_{corename}.fits')
	lam_model, flux_model = f2[1].data['wave_model'], f2[1].data['f_model_50']*float(f2[1].header['SCALE'])
	flux_model = gf.flam_to_fnu(lam_model, flux_model) * (1.e23) * (1.e9)
	
	flux_model = np.interp(lam, lam_model, flux_model)
	
	if units=='fnu':
		lam = lam / 10000.
	elif units=='flam':
		flux = gf.fnu_to_flam(lam, flux*(1.e-23)*(1.e-9))
		flux_model = gf.fnu_to_flam(lam, flux_model*(1.e-23)*(1.e-9))
		
	return lam, flux, flux_model

def plot_composite(filename, plot_gsf=True, plot_emission=True, frame='obs', dust_corr=False, xlims=None, ylims=None, plot_lines=None):
	from astropy.io import ascii
	from astropy.io import fits as pyfits
	import matplotlib.pyplot as plt
	from PyGRB import general_functions as gf
	import numpy as np
	
	def gauss(x, x0, A, sig):
		g = A * np.exp(-0.5*(((x-x0)**2.)/(2*sig**2.)))
		return g
	
	basename = filename.split('/')[-1]
	corename = ''
	for i,part in enumerate(basename.split('_')):
		if i==(len(basename.split('_'))-1):
			continue
		corename += part
		corename += '_'
	corename = corename[0:-1]
	
	z_stack = float(corename[1:].split('_')[0].replace('p','.'))
		
	
	f = ascii.read(f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/input_stacks/{filename}.txt')
	fmod = pyfits.open(f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/output_stacks/gsf_spec_{basename}.fits')
	f['lambda(AA)'] = f['lambda(AA)'] / 10000.
	lam0 = (f['lambda(AA)'] / (1. + z_stack)) * 10000.

	fSFH = pyfits.open(f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/output_stacks/SFH_{basename}.fits')
	print('GSF properties')
	print('--------------')
	print('log Stellar mass [Mstar/Msol] : %0.3f'%float(fSFH[0].header['Mstel_50']))
	print('SFRUV [Msol/yr] : %0.3f'%float(fmod[1].header['SFRUV50']))
	print('log sSFR [yr^-1] : %0.3f'%(np.log10(float(fmod[1].header['SFRUV50'])/(10**float(fSFH[0].header['Mstel_50'])))))
	print('Z [Zsol] : %0.3f'%(10**float(fSFH[0].header['Z_LW_50'])))
	print('t_age [Myrs] : %0.3f'%((10**float(fSFH[0].header['T_LW_50']))*1000.))
	print('Av [mag] : %0.1f'%float(fSFH[0].header['AV0_50']))
	
	fig = plt.figure(figsize=(8.,4.5))
	ax = fig.add_subplot(111)
	if frame=='obs':
		ax.step(f['lambda(AA)'], f['flux_nu'], linewidth=1., color='black')
		ax.set_xlim(min(f['lambda(AA)']), max(f['lambda(AA)']))
	elif frame=='rest':
		ax.step(lam0, f['flux_nu'], linewidth=1., color='black')
		ax.set_xlim(min(lam0), max(lam0))
		
	if plot_gsf==True:
		fmod_wave = fmod[1].data['wave_model']
		fmod_flam = fmod[1].data['f_model_50'] * float(fmod[1].header['SCALE'])

		fmod_flux = gf.flam_to_fnu(fmod_wave, fmod_flam) * (1.e23) * (1.e9)
		fmod_wave = fmod_wave / 10000.
		fmod_lam0 = (fmod_wave / (1.+z_stack)) * 10000.

		fmod_flux = np.interp(f['lambda(AA)'], fmod_wave, fmod_flux)
		fmod_flam = np.interp(f['lambda(AA)'], fmod_wave, fmod_flam)
		fmod_wave = f['lambda(AA)'].copy()
		fmod_lam0 = lam0.copy()


		if frame=='obs':
			ax.step(fmod_wave, fmod_flux, linewidth=1.5, color='red', zorder=100)
		elif frame=='rest':
			ax.step(fmod_lam0, fmod_flux, linewidth=1.5, color='red', zorder=100)
		
		if plot_emission==True:
			print()
			print('Line  Lam_rest [Ang]  Lam_obs [Ang]  A [x 1E-21 cgs]  Sig [Ang]  Line Flux [x 1E-21 cgs]  EW0 [A]')
			print('-----------------------------------------------------------------------------------------------------------------')
			# if dust_corr==False:
			fem = pyfits.open(f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/emission_fits_individual/{basename}.fits')
			# elif dust_corr==True:
			# 	fem = pyfits.open(f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/emission_fits_individual/{basename}_dustcorr.fits')
				
			lines = []
			for key in fem[1].header.keys():
				if ('SIGMA_ERR' in key):
					lines.append(key[10:])
	
			total = fmod_flux.copy()
			for line in lines:
				lam_rest = float(fem[1].header[f'LAM0_{line}'])
				x0 = float(fem[1].header[f'LAM_{line}'])
				A = float(fem[1].header[f'A_{line}'])
				sig = float(fem[1].header[f'SIGMA_{line}'])

				prof_flam = gauss(fmod_wave*10000., x0, A, sig) * (1.e-21)
				prof = gf.flam_to_fnu(fmod_wave*10000., prof_flam) * (1.e23) * (1.e9)
				total = total + prof

				line_flux = ufloat(float(fem[1].header[f'FLUX_{line}']), float(fem[1].header[f'FLUX_ERR_{line}']))
				EW0 = ufloat(float(fem[1].header[f'EW0_{line}']), float(fem[1].header[f'EW0_ERR_{line}']))
				print(line, '%0.1f'%lam_rest, '%0.1f'%x0, '%0.1f'%A, '%0.1f'%sig, '%0.3f+/-%0.3f'%(line_flux.n,line_flux.s), '%0.3f+/-%0.3f'%(EW0.n,EW0.s))

				if frame=='obs':
					ax.plot([x0/10000.,x0/10000.], ax.set_ylim(), color='black', linestyle='--')
					ax.plot(fmod_wave, fmod_flux+prof, color='darkorange')
					ax.text(x0/10000., max(fmod_flux + prof), s=line, color='green', ha='center', va='bottom')
				elif frame=='rest':
					ax.plot([(x0/10000.)/(1.+z_stack),(x0/10000.)/(1.+z_stack)], ax.set_ylim(), color='black', linestyle='--')
					ax.plot(fmod_lam0, fmod_flux+prof, color='darkorange')
					ax.text(x0/(1.+z_stack), max(fmod_flux + prof), s=line, color='green', ha='center', va='bottom')

			if frame=='obs':
				ax.plot(fmod_wave, total, color='dodgerblue')
			elif frame=='rest':
				ax.plot(fmod_lam0, total, color='dodgerblue')

	if xlims!=None:
		ax.set_xlim(xlims[0],xlims[1])
	if ylims!=None:
		ax.set_ylim(ylims[0],ylims[1])

	if plot_lines!=None:
		for line in plot_lines:
			if frame=='obs':
				line_z = line*(1.+z_stack) / 10000.
			else:
				line_z = line
			ax.plot([line_z,line_z], ax.set_ylim(), linestyle='--', color='grey')

	if frame=='obs':
		gf.style_axes(ax, 'Observed Wavelength [$\mu$m]', 'Flux Density [nJy]')
	elif frame=='rest':
		gf.style_axes(ax, 'Rest Wavelength [$\AA$]', 'Flux Density [nJy]')
	plt.tight_layout()
	plt.show()

def correct_slitloss(spec_lam, spec_flux):
	"""
	spec_lam : the wavelength array of the spectrum to be corrected, in microns
	spec_flux : the flux array of the spectrum to be corrected, in any Fnu units
	"""
	sl = ascii.read('/Users/guidorb/Dropbox/papers/z5_Templates/slitloss_percentage.txt')
	sl_lam, sl_perc = sl['wave'], sl['percent']
	
	sl_perc = np.interp(spec_lam, sl_lam, sl_perc)
	corrected_spec = spec_flux / (1. - (sl_perc/100.))
	return corrected_spec

def MUV_from_spec(ilam, iflux, z, mu=1.):
	## ilam : array of observed wavelengths, in microns
	## iflux : array of fluxes, in micro-jy
	## z : redshift of source
	## mu : magnification factor of source
	ilam_rest = ilam / (1.+z)
	fUV = np.nanmedian(iflux[(ilam_rest > 0.15) & (ilam_rest < 0.2)]) * (1.e-6)
	fUV_err = np.nanstd(iflux[(ilam_rest > 0.15) & (ilam_rest < 0.2)]) * (1.e-6)
	if fUV <= 0.:
		return 99.
	
	mUV = gf.flux_to_AB(fUV, flux_err=fUV_err, unit='jy')
	
	DL = cosmo.luminosity_distance(z).to(u.pc).value
	MUV = mUV - 5.*unp.log10(DL) + 5. + 2.5*unp.log10(1+z) + 2.5*unp.log10(mu)
	return MUV

def MUV_from_photo(fUV, z, fUV_err=None, mu=1.):
	## fUV : UV flux in micro-jy
	## fUV_err : UV flux error in micro-jy
	## z : redshift of source
	## mu : magnification factor
	DL = cosmo.luminosity_distance(z).to(u.pc).value
	if fUV_err is not None:
		mUV = gf.flux_to_AB(fUV*(1.e-6), flux_err=fUV_err*(1.e-6), unit='jy')
		MUV = mUV - 5.*unp.log10(DL) + 5. + 2.5*unp.log10(1+z) + 2.5*unp.log10(mu)
	else:
		mUV = gf.flux_to_AB(fUV*(1.e-6), unit='jy')
		MUV = mUV - 5.*np.log10(DL) + 5. + 2.5*np.log10(1+z) + 2.5*np.log10(mu)
	return MUV

def R_Sanders(R, Rerr=None, Rtype='R23', xlims=[7.,8.01]):
	# R23 = ([OIII]4960,5008 + [OII]3727,3730) / Hb
	# returns log(O/H) for a given R23 value (not in log), assuming that values do not go over the turnover.
	# An errorbar of 0.2 dex should be assumed for the returned value.
	if Rtype=='R23':
		p = [-0.331,0.026,1.017]
	elif Rtype=='O32':
		p = [-1.153,0.723]
	elif Rtype=='NE3O2':
		p = [-0.998,-0.386]
	xvals = np.arange(xlims[0],xlims[1],0.01) - 8.
	yvals = np.polyval(p, xvals)
	
	if Rerr==None:
		logR = unp.log10(R)
	else:
		logR = unp.log10(ufloat(R,Rerr))

	metal = xvals[gf.find_nearest(yvals, unp.nominal_values(logR))] + 8.
	metal_high = xvals[gf.find_nearest(yvals, unp.nominal_values(logR)-unp.std_devs(logR))] + 8.
	metal_low = xvals[gf.find_nearest(yvals, unp.nominal_values(logR)+unp.std_devs(logR))] + 8.
	metal_err = np.mean([metal-metal_low,metal_high-metal])
	return ufloat(metal, abs(metal_err))

def calc_ebv(ratio):
	# dust law of Salim et al. (2018)
	Rv = 3.15

	lam = 0.6563
	Dlam = (1.57 * (lam**2.) * 0.35**2.) / (((lam**2.) - (0.2175**2.))**2 + ((lam**2)*0.35**2))
	kHa = -4.30 + 2.71*lam**-1 - 0.191*lam**-2 + 0.0121*lam**-3 + Dlam + Rv

	lam = 0.4861
	Dlam = (1.57 * (lam**2.) * 0.35**2.) / (((lam**2.) - (0.2175**2.))**2 + ((lam**2)*0.35**2))
	kHb = -4.30 + 2.71*lam**-1 - 0.191*lam**-2 + 0.0121*lam**-3 + Dlam + Rv

	ebv = (2.5 / (kHb-kHa)) * np.log10(ratio/2.86)
	return ebv

def bin_2d_spec(spec2d, factor=1, axis='lam'):
	factor = int(factor)
	
	def submatsum(data,n,m):
		# return a matrix of shape (n,m)
		bs = data.shape[0]//n,data.shape[1]//m  # blocksize averaged over
		return np.reshape(np.array([np.sum(data[k1*bs[0]:(k1+1)*bs[0],k2*bs[1]:(k2+1)*bs[1]]) for k1 in range(n) for k2 in range(m)]),(n,m))

	m, n = np.shape(spec2d)
	if axis=='lam':
		new_spec2d = submatsum(spec2d, m, int(n/factor))
	elif axis=='spatial':
		new_spec2d = submatsum(spec2d, int(m/factor), n)
	elif axis=='all':
		new_spec2d = submatsum(spec2d, int(m/factor), int(n/factor))
	return new_spec2d

def bin_1d_spec(lam, spec1d, err1d=None, factor=1, method='median'):
	factor = int(factor)
	n      = len(lam)
	n_full = n // factor
	trunc  = n_full * factor

	if method == 'sum':
		_reduce = np.nansum
	elif method == 'median':
		_reduce = np.nanmedian
	else:
		_reduce = np.nanmean

	# Vectorized reduction over complete bins
	ilam  = np.nanmedian(lam[:trunc].reshape(n_full, factor), axis=1)
	ispec = _reduce(spec1d[:trunc].reshape(n_full, factor), axis=1)
	if err1d is not None:
		ierr = np.sqrt(_reduce(err1d[:trunc].reshape(n_full, factor)**2., axis=1))

	# Partial last bin (when n is not a multiple of factor)
	if trunc < n:
		ilam  = np.append(ilam,  np.nanmedian(lam[trunc:]))
		ispec = np.append(ispec, _reduce(spec1d[trunc:]))
		if err1d is not None:
			ierr = np.append(ierr, np.sqrt(_reduce(err1d[trunc:]**2.)))

	if err1d is not None:
		return ilam, ispec, ierr
	return ilam, ispec

def _plot_emission_lines(ax, z, lines=None, frame='obs'):
	"""Overplot emission line markers on an existing spectrum axis."""
	if lines is None:
		lines = _EMISSION_LINES
	ylims = ax.get_ylim()
	midpoint  = ylims[0] + (ylims[1] - ylims[0]) * 0.6
	highpoint = ylims[0] + (ylims[1] - ylims[0]) * 0.7
	textpoint = ylims[0] + (ylims[1] - ylims[0]) * 0.725
	xlims = ax.get_xlim()
	for line, wavs in lines.items():
		last_lz = None
		for wav in wavs:
			lz = (wav / 10000.) * (1. + z) if frame == 'obs' else wav
			if xlims[0] <= lz <= xlims[1]:
				ax.plot([lz, lz], [midpoint, highpoint], linestyle='--', color='darkgray')
				last_lz = lz
		if last_lz is None:
			continue
		label = line.split('_')[0]
		offsets = {'[Ne III]': 0.05, '[O III]': 0.08, r'H$\alpha$': -0.065}
		ax.text(last_lz + offsets.get(label, 0.), textpoint, va='bottom', ha='center',
				color='darkgray', s=label, rotation=90, fontsize=10.)


def plot_prism_spectrum(spectra, msaid, frame='obs', clim=(-0.04,0.09), ylims=None, xlims=None, plot_model=False, mask_lines=False, plot_clipped=False, plot_lines=True, lines_to_plot=None):
	fcat = ascii.read('/Users/guidorb/Dropbox/Catalogs/JEWELS/highz_msaid_full.dat')
	
	R = 'prism-clear' # 'prism' or 'grism'
	
	itab = np.where(fcat['msaid'] == msaid)[0]
	z_spec = fcat['z'][itab]
	
	lam, flux, err = spectra[msaid][R]['lam'], spectra[msaid][R]['flux'], spectra[msaid][R]['err']
	lam_A, lam0 = lam * 10000., (lam * 10000.) / (1.+z_spec)
	if 'mask' not in spectra[msaid][R]:
		mask = np.zeros_like(lam)
	else:
		mask = spectra[msaid][R]['mask']
		
	if frame=='rest':
		lam = lam0.copy()
		
	if mask_lines==True:
		for line in _MASK_LINES:
			dv = 10000. if line == r'Ly$\alpha$' else 1000.
			for lam_cent in _MASK_LINES[line]:
				zcent = lam_cent * (1. + z_spec)
				deltalam = (dv / _C_KM_S) * zcent
				if line == r'Ly$\alpha$':
					zlam_min, zlam_max = zcent - deltalam, 1500. * (1. + z_spec)
				else:
					zlam_min, zlam_max = zcent - deltalam, zcent + deltalam
				mask[(lam_A >= zlam_min) & (lam_A <= zlam_max)] = 1
	mask = mask.astype(int)
	im = spectra[msaid][R]['2D']
	
	fig = plt.figure(figsize=(10.,6.))
	gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[0.25,0.75])
	ax1 = fig.add_subplot(gs[0])
	ax1.pcolormesh(lam, np.arange(np.shape(im)[0]), im, cmap='magma', clim=clim)
	ax1.set_ylim(0,np.shape(im)[0])
	ycent = int(np.shape(im)[0]/2)
	ax1.set_ylim(ycent-10,ycent+10)
	ax1.set_xlim(min(lam), max(lam))
	gf.style_axes(ax1)
	ax1.set_xticklabels([])
	ax1.set_yticks([])
	ax1.set_yticklabels([])
	
	ax2 = fig.add_subplot(gs[1])
	if plot_clipped==True:
		masked = np.ma.masked_array(flux, mask=mask)
		values = sigma_clip(masked, sigma=3., maxiters=5, cenfunc='median', stdfunc='std', masked=True)
		values.mask[(masked.mask == True)] = True
		mask[(values.mask==True)] = 1
	
	ax2.step(lam, flux, color='C0', linewidth=1.25)
	ax2.set_xlim(ax1.set_xlim())
	
	if plot_clipped==True:
		ax2.step(lam, values, color='turquoise', linewidth=1.5)
	
	if plot_model==True:
		mod = pyfits.open(f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/output/gsf_spec_{msaid}.fits')
		lam_mod, flux_mod = mod[1].data['wave_model'], mod[1].data['f_model_noline_50']*(1.e-19)
		flux_mod = gf.flam_to_fnu(lam_mod, flux_mod) * (1.e23) * (1.e6)
		lam_mod = lam_mod / 10000.
		model = np.interp(lam, lam_mod, flux_mod)
		ax2.plot(lam, model, color='darkred', linewidth=2.)
	
	ax2.plot(ax2.set_xlim(), [0.,0.], linestyle='--', color='black')
	if frame=='obs':
		gf.style_axes(ax2, r'Observed Wavelength [$\mu$m]', r'Flux Density [$\mu$Jy]')
	elif frame=='rest':
		gf.style_axes(ax2, r'Rest Wavelength [$\mu$m]', r'Flux Density [$\mu$Jy]')
	
	if ylims==None:
		low = np.nanmin(flux[(lam < 1.) & (lam > 0.8)])
		if low >= 0.:
			low = -low
		low = low + (low * 1.75)
		
		high = np.nanmax(flux[(lam > 1.) & (lam < 2.)])
		high = high + (high * 3.)
		
		if (np.isfinite(low)==True) & (np.isfinite(high)==True):
			ax2.set_ylim(low, high)
		else:
			ax2.set_ylim()
	else:
		ax2.set_ylim(ylims[0],ylims[1])
		
	if xlims==None:
		ax2.set_xlim(ax2.set_xlim())
	else:
		ax1.set_xlim(xlims[0], xlims[1])
		ax2.set_xlim(xlims[0], xlims[1])

	ax2.fill_between(lam, np.ones_like(lam)*ax2.set_ylim()[0], np.ones_like(lam)*ax2.set_ylim()[1], color='silver', where=(mask==1))
	
	if ('photo' in spectra[msaid]) & (frame=='obs'):
		for ip in range(len(spectra[msaid]['photo']['filters'])):
			lam_cent, lam_err, _ = gf.get_filter_info(spectra[msaid]['photo']['filters'][ip].split('/')[-1], output_unit='mu')
			snr = spectra[msaid]['photo']['flux_aper'][ip] / spectra[msaid]['photo']['flux_aper_err'][ip]
			
			if snr > 2.:
				ax2.errorbar(lam_cent, spectra[msaid]['photo']['flux_aper'][ip], xerr=[[lam_err[0]],[lam_err[1]]], yerr=spectra[msaid]['photo']['flux_aper_err'][ip], marker='o', ecolor='navy', color='royalblue', markeredgecolor='navy', capsize=2.)
			else:
				ax2.errorbar(lam_cent, abs(spectra[msaid]['photo']['flux_aper'][ip])+spectra[msaid]['photo']['flux_aper_err'][ip], xerr=[[lam_err[0]],[lam_err[1]]], yerr=[25.], uplims=[1], ecolor='navy', capsize=2.)

			fl = gf.photo_from_filter(lam, flux, filt=spectra[msaid]['photo']['filters'][ip].split('/')[-1])
			ax2.errorbar(lam_cent, fl, marker='s', color='white', markeredgecolor='darkred', markeredgewidth=2.)

	ax2.annotate(xy=(0.025,0.9), xycoords=('axes fraction'), text=msaid+r', $z_{\rm spec.}=%0.3f$'%(z_spec), fontsize=15)
	
	if plot_lines==True:
		_plot_emission_lines(ax2, z_spec, lines_to_plot if lines_to_plot is not None else _EMISSION_LINES, frame=frame)

	plt.tight_layout()
	plt.show()
	
	if plot_model==False:
		return lam, flux, err, mask
	else:
		return lam, flux, err, mask, model


def plot_msaexp_spectrum(msaid_file, clim=(-0.04,0.09), ylims=None, xlims=None, plot_lines=True, z=None):
	f = pyfits.open(msaid_file)
	lam, flux, err = f[1].data['wave'], f[1].data['flux'], f[1].data['err']
	im = f[0].data
	f.close()

	if z is None:
		try:
			fz = pyfits.open(msaid_file.replace('.fits', 'zfit.fits'))
			z = fz[0].header['z']
			fz.close()
		except Exception:
			z = None

	if plot_lines:
		assert z is not None, 'Need to specify redshift to use for line plotting...'

	msaid = msaid_file.split('/')[-1].replace('.fits', '')

	fig = plt.figure(figsize=(10., 6.))
	gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[0.25, 0.75])
	ax1 = fig.add_subplot(gs[0])
	ax1.pcolormesh(lam, np.arange(np.shape(im)[0]), im, cmap='magma', clim=clim)
	ycent = int(np.shape(im)[0] / 2)
	ax1.set_ylim(ycent - 10, ycent + 10)
	ax1.set_xlim(min(lam), max(lam))
	gf.style_axes(ax1)
	ax1.set_xticklabels([])
	ax1.set_yticks([])
	ax1.set_yticklabels([])

	ax2 = fig.add_subplot(gs[1])
	ax2.step(lam, flux, color='C0', linewidth=1.25)
	ax2.set_xlim(ax1.get_xlim())
	ax2.plot(ax2.get_xlim(), [0., 0.], linestyle='--', color='black')
	gf.style_axes(ax2, r'Observed Wavelength [$\mu$m]', r'Flux Density [$\mu$Jy]')

	if ylims is None:
		valid_low = flux[(lam < 1.) & (lam > 0.8)]
		valid_high = flux[(lam > 1.) & (lam < 2.)]
		low = np.nanmin(valid_low) if valid_low.size else 0.
		if low >= 0.:
			low = -low
		low = low + low * 1.75
		high = np.nanmax(valid_high) if valid_high.size else 1.
		high = high + high * 3.
		if np.isfinite(low) and np.isfinite(high):
			ax2.set_ylim(low, high)
	else:
		ax2.set_ylim(ylims[0], ylims[1])

	if xlims is not None:
		ax1.set_xlim(xlims[0], xlims[1])
		ax2.set_xlim(xlims[0], xlims[1])

	ax2.annotate(xy=(0.025, 0.9), xycoords='axes fraction',
				 text=msaid + (r', $z_{\rm spec.}=%0.3f$' % z if z is not None else ''),
				 fontsize=15)

	if plot_lines and z is not None:
		_plot_emission_lines(ax2, z, _EMISSION_LINES)

	plt.tight_layout()
	plt.show()



def plot_grating_spectrum(spectra, msaid, frame='obs', clim=(-0.04,0.09), ylims=None, xlims=None, plot_model=False, mask_lines=False, plot_clipped=False, plot_lines=True, units='fnu'):
	fcat = ascii.read('/Users/guidorb/Dropbox/Catalogs/JEWELS/highz_msaid_full.dat')
	
	itab = np.where(fcat['msaid'] == msaid)[0]
	z_spec = fcat['z'][itab]
	
	lam, flux, err = [], [], []
	for R in spectra[msaid].keys():
		if R in ['prism-clear']:
			continue
		lam = np.concatenate([lam, spectra[msaid][R]['lam']])
		flux = np.concatenate([flux, spectra[msaid][R]['flux']])
		err = np.concatenate([flux, spectra[msaid][R]['err']])
	i = np.argsort(lam)
	lam = lam[i]
	flux = flux[i]
	err = err[i]

	lam_A, lam0 = lam * 10000., (lam * 10000.) / (1.+z_spec)
	
	fig = plt.figure(figsize=(10.,5.))	
	ax2 = fig.add_subplot(111)
	if frame=='obs':
		ax2.step(lam, flux, color='C0', linewidth=1.25)
		ax2.set_xlim(min(lam), max(lam))
		gf.style_axes(ax2, r'Observed Wavelength [$\mu$m]', r'Flux Density [$\mu$Jy]')
	elif frame=='rest':
		ax2.step(lam0, flux, color='C0', linewidth=1.25)
		ax2.set_xlim(min(lam0), max(lam0))
		gf.style_axes(ax2, r'Rest Wavelength [$\mu$m]', r'Flux Density [$\mu$Jy]')
	
	ax2.plot(ax2.set_xlim(), [0.,0.], linestyle='--', color='black')
	
	
	if ylims is None:
		ax2.set_ylim(ax2.get_ylim())
	else:
		ax2.set_ylim(ylims[0], ylims[1])

	if xlims is not None:
		ax2.set_xlim(xlims[0], xlims[1])

	ax2.annotate(xy=(0.025,0.9), xycoords=('axes fraction'), text=msaid+r', $z_{\rm spec.}=%0.3f$'%(z_spec), fontsize=15)

	if plot_lines==True:
		_plot_emission_lines(ax2, z_spec, _EMISSION_LINES_GRATING, frame=frame)

	plt.tight_layout()
	plt.show()

	return lam, flux, err
	
	
def get_prism_spectrum_backup(spectra, target, mask_lines=False, clip=False, sigma=3., get_model=False, demagnify=False, correct_slitloss=False):
	
	# tab = ascii.read('/Users/guidorb/Dropbox/papers/z5_Templates/highz_galaxies_lit.dat')
	tab = ascii.read('/Users/guidorb/Dropbox/Postdoc/highz_galaxies_combined.dat')
	
	R = 'prism' # 'prism' or 'grism'
	
	itab = np.where(tab['name'] == target)[0]
	z_spec = tab['z_spec'][itab]
	if demagnify==True:
		mu = tab['mu'][itab]
	
	lam, flux, err = spectra[target][R]['lam'], spectra[target][R]['flux'], spectra[target][R]['err']
	if 'mask' not in spectra[target][R]:
		mask = np.zeros_like(lam)
	else:
		mask = spectra[target][R]['mask']

	if mask_lines==True:
		for line in _MASK_LINES:
			dv = 10000. if line == r'Ly$\alpha$' else 1000.
			for lam_cent in _MASK_LINES[line]:
				zcent = lam_cent * (1. + z_spec)
				deltalam = (dv / _C_KM_S) * zcent
				if line == r'Ly$\alpha$':
					zlam_min, zlam_max = zcent - deltalam, 1500. * (1. + z_spec)
				else:
					zlam_min, zlam_max = zcent - deltalam, zcent + deltalam
				mask[gf.find_nearest(lam*10000., zlam_min):gf.find_nearest(lam*10000., zlam_max)+1] = 1
	mask = mask.astype(int)
	
	if 'photo' in spectra[target]:
		ip = np.where(spectra[target]['photo']['filters'] == ('/Users/guidorb/Dropbox/Postdoc/filters/'+spectra[target]['photo']['normfilt']))[0][0]
		photo = spectra[target]['photo']['flux_aper'][ip]

		if demagnify==True:
			photo = gf.flux_to_AB(gf.AB_to_flux(photo)/mu)

		spec_photo = gf.photo_from_filter(lam, flux, filt=spectra[target]['photo']['normfilt'])
		ratio = photo/spec_photo
		flux = flux * ratio
		err = err * ratio

		if correct_slitloss==True:
			slitloss = []
			for ip in range(len(spectra[target]['photo']['filters'])):
				p = gf.find_nearest(lam, spectra[target]['photo']['lam_cent'][ip])
				fl = gf.photo_from_filter(lam, flux, filt=spectra[target]['photo']['filters'][ip].split('/')[-1])

				if lam[p] > 1.45:
					slitloss.append([lam[p], spectra[target]['photo']['flux_aper'][ip] / fl])
			slitloss = np.array(slitloss)

			poly = np.polyfit(slitloss.T[0], slitloss.T[1], 2)
			yslitloss = np.polyval(poly, lam)
			yslitloss[(lam < 1.45)] = 1.
			flux = flux * yslitloss
	
	if clip==True:
		masked = np.ma.masked_array(flux, mask=mask)
		values = sigma_clip(masked, sigma=sigma, maxiters=5, cenfunc='median', stdfunc='std', masked=True)
		values.mask[(masked.mask == True)] = True
		mask[(values.mask==True)] = 1

	if get_model==True:
		# mod = pyfits.open(f'/Volumes/GRBHD/JWST/gsf/output/gsf_spec_{target}.fits')
		mod = pyfits.open(f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/output/gsf_spec_{target}.fits')
		lam_mod, flux_mod = mod[1].data['wave_model'], mod[1].data['f_model_noline_50']*(1.e-19)
		flux_mod = gf.flam_to_fnu(lam_mod, flux_mod) * (1.e23) * (1.e6)
		lam_mod = lam_mod / 10000.
		model = np.interp(lam, lam_mod, flux_mod)
		return lam, flux, err, mask, model
	else:
		return lam, flux, err, mask


def get_grating_spectrum(spectra, target, filters=['F100LP','F170LP','F290LP'], plot=False, xlims=None, ylims=None):
	
	tab = ascii.read('/Users/guidorb/Dropbox/papers/z5_Templates/highz_galaxies_lit.dat')
	
	R = 'grating' # 'prism' or 'grism'
	
	itab = np.where((tab['prog_id'] == int(target.split('_')[0])) & 
					(tab['id'] == int(target.split('_')[1])))[0]
	z_spec = tab['z_spec'][itab]
	
	if plot==True:
		fig = plt.figure(figsize=(8.,5.))
		ax = fig.add_subplot(111)
	
	ilam = []
	iflux = []
	ierr = []
	for grating in spectra[target][R].keys():
		if (grating.split('-')[0].upper() not in filters):
			continue
			
		lam, flux, err = spectra[target][R][grating]['lam'], spectra[target][R][grating]['flux'], spectra[target][R][grating]['err']
		ilam = np.concatenate([ilam, lam])
		iflux = np.concatenate([iflux, flux])
		ierr = np.concatenate([ierr, err])
		
		if plot==True:
			ax.step(lam, flux, linewidth=1.5)
		
	idx = np.argsort(ilam)
	ilam = ilam[idx]
	iflux = iflux[idx]
	ierr = ierr[idx]
	
	if plot==True:
		if ylims is not None:
			ax.set_ylim(ylims[0], ylims[1])
		if xlims is not None:
			ax.set_xlim(xlims[0], xlims[1])
		else:
			ax.set_xlim(min(ilam), max(ilam))
		ax.plot([1.0727,1.0727], ax.set_ylim(), linestyle='--', color='darkred')
		gf.style_axes(ax, r'Observed Wavelength [$\mu$m]', r'Flux Density $\mu$Jy', fontsize=17.5, labelsize=17.5)
		plt.tight_layout()
		plt.show()

	return ilam, iflux, ierr


def custom_sigma_clip(array, low=3., high=3., op='median'):
	if op=='mean':
		mu = np.nanmean(array)
	elif op=='median':
		mu = np.nanmedian(array)
	sig = np.nanstd(array)
	lowval = mu-sig
	highval = mu+sig
	idx = np.where((array >= lowval) &
				   (array <= highval))[0]
	return array[idx], lowval, highval, idx


def calibrate_etc_spec(file, mode, add_noise=True, return_cals=False, noise_scale=1., boost_flux=1.):

	# 'PRISM_FLAM'
	# 'PRISM_FNU'
	# 'G140M_F100LP_FLAM'
	# 'G140M_F100LP_FNU'
	# 'G235M_F170LP_FNU'
	# 'G395M_F290LP_FNU'
	# 'MIRI_LRS_FNU'
	# 'MIRI_LRS_FLAM'

	import pickle
	from astropy.io import fits as pyfits
	
	calibrations = pickle.load(open('/Users/guidorb/Dropbox/Admin/Proposals/ETC_Flux_Calibrations/ETC_flux_calibrations.p', "rb"), encoding='latin1')
	cal_lam, cal_resp = calibrations[mode]['lam'], calibrations[mode]['cal']
	
	if file[-1]=='/':
		file = file[0:-1]
	f = pyfits.open(f'{file}/lineplot/lineplot_extracted_flux.fits')
	ilam, ispec = f[1].data['WAVELENGTH'], f[1].data['extracted_flux']
	cal_resp = np.interp(ilam, cal_lam, cal_resp)

	f = pyfits.open(f'{file}/lineplot/lineplot_sn.fits')
	isnr = f[1].data['sn']

	ispec *= cal_resp
	
	ispec *= ((1.e6) * boost_flux)# to nJy
	ierr = (ispec/isnr)
	if add_noise==True:
		ispec += np.random.normal(loc=np.zeros_like(ispec), scale=ierr*noise_scale)
	
	if return_cals==False:
		return ilam, ispec, ierr, isnr
	else:
		return ilam, ispec, ierr, isnr, cal_lam, cal_resp


def stack_spectra(stack_flux, stack_err=None, op='median', clip=False, sigma_sig=3.):
	# stack_flux : 2D ARRAY of stacked fluxes, in a common reference frame
	# stack_err  : 2D ARRAY of stacked uncertainties (standard deviations)

	if isinstance(stack_flux, list):
		stack_flux = np.array(stack_flux)
	if stack_err is not None:
		if isinstance(stack_err, list):
			stack_err = np.array(stack_err)

	base_mask = (stack_flux != 0.) & np.isfinite(stack_flux)
	if stack_err is not None:
		base_mask &= (stack_err != 0.) & np.isfinite(stack_err)

	flux_m = np.where(base_mask, stack_flux, np.nan)

	if clip:
		mu  = np.nanmedian(flux_m, axis=0) if op == 'median' else np.nanmean(flux_m, axis=0)
		sig = np.nanstd(flux_m, axis=0)
		clip_mask = (flux_m >= mu - sig) & (flux_m <= mu + sig)
		valid = base_mask & clip_mask
	else:
		valid = base_mask

	flux_m = np.where(valid, stack_flux, np.nan)
	nstack = valid.sum(axis=0).astype(float)

	if op == 'median':
		iflux = np.nanmedian(flux_m, axis=0)
	else:
		iflux = np.nanmean(flux_m, axis=0)

	std_vals = np.nanstd(flux_m, ddof=1, axis=0)
	istd = np.where(nstack > 1, std_vals / np.sqrt(nstack), 0.)

	if stack_err is not None:
		err_m  = np.where(valid, stack_err, np.nan)
		n_safe = np.where(nstack > 0, nstack, 1.)
		ierr   = np.sqrt(np.nansum(err_m**2, axis=0)) / n_safe
		return iflux, istd, ierr, nstack
	return iflux, istd, nstack

def generate_line_mask(ilam, z_spec=0., dv=2000):
	"""
	ilam : array of observed wavelengths in Angstroms
	z_spec : redshift of the source
	dv : velocity width in km/s to mask around each line
	"""
	mask = np.zeros_like(ilam)
	lam_min, lam_max = np.nanmin(ilam), np.nanmax(ilam)

	for line, wavs in _MASK_LINES.items():
		for lam_cent in wavs:
			zcent = lam_cent * (1. + z_spec)
			if (zcent < lam_min) or (zcent > lam_max):
				continue
			deltalam = (dv / _C_KM_S) * zcent
			if line == r'Ly$\alpha$':
				zlam_min, zlam_max = zcent, 1500. * (1. + z_spec)
			else:
				zlam_min, zlam_max = zcent - deltalam, zcent + deltalam
			mask[gf.find_nearest(ilam, zlam_min):gf.find_nearest(ilam, zlam_max) + 1] = 1

	return mask.astype(int)



def plot_line_fitting_results(root=None):
	
	import matplotlib.pyplot as plt
	from matplotlib import gridspec
	from PyGRB import general_functions as gf
	import numpy as np
	from astropy.io import ascii
	from dust_extinction.parameter_averages import G23
	import os
	
	lines = {'HEII_1':1640.420,
			 'OIII_05':1663.4795,
			 'CIII':1908.734,
			 
			 'OII_UV_1':3727.092,
			 'OII_UV_2':3729.875,
			 'NEIII_UV_1':3869.86,
			 'NEIII_UV_2':3968.59,
			 'HDELTA':4102.8922,
			 'HGAMMA':4341.6837,
			 'OIII_1':4364.436,
			 'HEI_1':4471.479,
			 
			 'HEII_2':4685.710,
			 'HBETA':4862.6830,
			 'OIII_2':4960.295,
			 'OIII_3':5008.240,
			 
			 'HEI':5877.252,
			 
			 'HALPHA':6564.608,
			 'SII_1':6718.295,
			 'SII_2':6732.674}
	
	def gauss(x, x0, A, sig, off):
		g = A * unp.exp(-0.5*(((x-x0+off)**2.)/(2*sig**2.)))
		return g
	
	fig = plt.figure(figsize=(5.*5.,10.))
	gs = gridspec.GridSpec(ncols=5, nrows=2)
	
	# plot full spectrum first
	filename = f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/input_stacks/{root}_full.txt'
	f = ascii.read(filename)
	ilam, iflux, ierr = f['lambda(AA)'], f['flux_nu'], f['eflux_nu_lines']
	z_spec = float(root.split('z')[-1].split('_')[0].replace('p','.'))
	ilam0 = ilam / (1.+z_spec)
	ilam = ilam / 10000.
	
	norm = np.median(iflux[(ilam0 > 1500.) & (ilam0 < 2000.)])
	iflux /= norm
	ierr /= norm
	
	ax = fig.add_subplot(gs[0,:])
	ax.step(ilam, iflux, linewidth=3., zorder=2)
	ax.set_ylim()
	ax.fill_between(ilam, iflux-ierr, iflux+ierr, alpha=0.2, zorder=1)
	ax.set_xlim(0.6,5.3)
	ax.set_ylim(-0.75,6.)
	ax.plot(ax.set_xlim(), [0.,0.], linestyle='--', color='black', linewidth=1.)
	gf.style_axes(ax, r'Observed Wavelength [$\mu$m]', r'Flux Density [nJy]', fontsize=35, labelsize=35)
	
	if '/' in root:
		model_root = root.split('/')[-1]
	else:
		model_root = root
	
	filename = f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/output_stacks/gsf_spec_{model_root}_full.fits'
	fmodel = pyfits.open(filename)
	lam, model = fmodel[1].data['wave_model'], fmodel[1].data['f_model_noline_50'] * float(fmodel[1].header['SCALE'])
	model = gf.flam_to_fnu(lam, model) * (1.e23) * (1.e9)
	
	imodel = np.interp(ilam, lam/10000., model)
	imodel /= norm
	ax.plot(ilam, imodel, color='darkred', linewidth=4.)

	# create and plot emission lines from best fit
	fem = pyfits.open(f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/emission_fits/{model_root}_emission_lines.fits')
	
	for N, ext in enumerate(['CIII','Balmer','OIIIHbeta','HeI','Halpha']):
		ax = fig.add_subplot(gs[1,N])
		gf.style_axes(ax, r'Observed Wavelength [$\mu$m]', r'Flux Density [nJy]', fontsize=25, labelsize=25)
		
		# skip if file doesn't exist (i.e., lines aren't present)
		filename = f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/input_stacks/{root}_{ext}.txt'
		if os.path.isfile(filename)==False:
			continue
			
		f = ascii.read(filename)
		ilam, iflux, ierr = f['lambda(AA)'], f['flux_nu'], f['eflux_nu_lines']
		ilam = ilam / 10000.
		ilam0 = (ilam * 10000.) / (1.+z_spec)
		iflux /= norm
		ierr /= norm
		
		# get continuum model
		filename = f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/output_stacks/gsf_spec_{model_root}_{ext}.fits'
		fmodel = pyfits.open(filename)
		lam, model = fmodel[1].data['wave_model'], fmodel[1].data['f_model_noline_50'] * float(fmodel[1].header['SCALE'])
		model = gf.flam_to_fnu(lam, model) * (1.e23) * (1.e9)
		imodel = np.interp(ilam, lam/10000., model)
		imodel /= norm
		
		# # correct flux and continuum by dust
		# xAv = 1./(ilam0/10000.)
		# SFH = pyfits.open(f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/output_stacks/SFH_{model_root}_full.fits')
		# Av = float(SFH[0].header['AV_50'])
			
		# ext = G23(Rv=3.1)
		# idx_av = np.where((xAv >= ext.x_range[0]) & (xAv <= ext.x_range[1]))[0]
		# iflux = iflux/ext.extinguish(xAv[idx_av], Av=Av)
		# imodel = imodel[idx_av]/ext.extinguish(xAv[idx_av], Av=Av)
		
		# plot flux
		ax.step(ilam, iflux, linewidth=2.5, zorder=2)
		ax.set_xlim(min(ilam), max(ilam))
		ax.fill_between(ilam, iflux-ierr, iflux+ierr, alpha=0.2, zorder=1)
		ax.set_ylim()
		ax.plot(ax.set_xlim(), [0.,0.], linestyle='--', color='black', linewidth=1.)
		
		lines_present = []
		for key in fem[1].header:
			if 'DUSTCORR' in key:
				continue
			if ('FLUX_ERR' in key):
				line = key.split('FLUX_ERR_')[-1]
				if line not in lines_present:
					lines_present.append(line)
		
		profile = np.zeros_like(ilam) + imodel
		for line in lines_present:
			lam_line = lines[line] * (1.+z_spec)
			
			prof = gauss(ilam*10000., lam_line, fem[1].header[f'A_{line}'], fem[1].header[f'SIGMA_{line}'], fem[1].header[f'OFFSET_{line}']) * (1.e-21)
			prof = gf.flam_to_fnu(ilam*10000., prof) * (1.e23) * (1.e9)

			profile = profile + (prof/norm)
			ax.plot([lam_line/10000.,lam_line/10000.], ax.set_ylim(), linestyle='--', color='silver', linewidth=2.)
			
		ax.plot(ilam, profile, color='darkorange', linewidth=4.)
		ax.plot(ilam, imodel, color='darkred', linewidth=4.)
	
	plt.tight_layout()
	plt.show()
