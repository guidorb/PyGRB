import glob
import numpy as np
from astropy.modeling import models, fitting, Fittable1DModel, Parameter
from astropy.visualization import astropy_mpl_style, simple_norm
from scipy.interpolate import interp1d, RegularGridInterpolator
from matplotlib.patches import Rectangle
from ipywidgets import interact
import ipywidgets as widgets
from astropy.stats import sigma_clip
from copy import copy
import astropy.units as u
import pickle
from . import general_functions as gf
from . import analysis_functions as af
from astropy.io import ascii
from matplotlib import gridspec
from uncertainties import ufloat
from uncertainties import unumpy as unp
from astropy.constants import c
import os
from astropy.io import fits as pyfits
import random
from astropy.cosmology import Planck15 as cosmo
import matplotlib.pyplot as plt
import corner
from scipy.optimize import minimize
import emcee
import astropy.units as u
from astropy import __version__ as asver
# import gsf
# print(gsf.__version__)
# from gsf.function import get_input,write_input
# from gsf.gsf import run_gsf_template
# from gsf.plot_sed import plot_sed
# from gsf.plot_sfh import plot_sfh
# from gsf.function import read_input
from scipy.optimize import curve_fit
import scipy
from astropy.modeling import models,fitting
import tqdm
import extinction

c = c.to(u.km/u.s).value

class LineFitting:
	def __init__(self, lam=None, flux=None, err=None, units='msaexp', z=None, msaid_prism=None):
		"""
		lam : Observed wavelengths [Microns or Angstroms, will be converted to Angstrom]
		flux : Observed fluxes [micro-Jansky]
		err : Same units as flux
		z : Redshift
		"""
		if msaid_prism==None:
			assert (lam is not None) & (flux is not None) & (err is not None) & (z is not None), 'Need to pass an input spectrum and redshift.'

			idx = np.where((np.isfinite(lam)==True) | (np.isfinite(flux)==True) | (np.isfinite(err)==True))[0]

			if units=='msaexp':
				self.lam = lam[idx] * 10000.
				flam = gf.fnu_to_flam(self.lam, flux[idx]*(1.e-6)*(1.e-23), fnu_err=err[idx]*(1.e-6)*(1.e-23))
				self.flux = unp.nominal_values(flam)
				self.err = unp.std_devs(flam)
			elif units in ['cgs','CGS','flam','FLAM']:
				self.lam = lam[idx]
				self.flux = flux[idx]
				self.err = err[idx]
			self.z = z
			self.lam0 = self.lam / (1.+self.z)
		else:
			jewels = gf.load_jewels()
			i = np.where(jewels.tab['msaid'] == msaid_prism)[0][0]
			if z==None:
				self.z = jewels.tab['z'][i]
			else:
				self.z = z

			idx = np.where((np.isfinite(jewels.spectra[msaid_prism]['prism-clear']['lam'])==True) | 
							(np.isfinite(jewels.spectra[msaid_prism]['prism-clear']['flux'])==True) | 
							(np.isfinite(jewels.spectra[msaid_prism]['prism-clear']['err'])==True))[0]

			self.lam = jewels.spectra[msaid_prism]['prism-clear']['lam'][idx] * 10000.
			flam = gf.fnu_to_flam(self.lam, jewels.spectra[msaid_prism]['prism-clear']['flux'][idx]*(1.e-6)*(1.e-23), 
								fnu_err=jewels.spectra[msaid_prism]['prism-clear']['err'][idx]*(1.e-6)*(1.e-23))
			self.flux = unp.nominal_values(flam)
			self.err = unp.std_devs(flam)
			self.lam0 = self.lam / (1.+self.z)

	def fit_single(self, lam_window=100., normval=1., verbose=True, plot_results=False, lines_to_fit=None, fit_params={}):
		"""
		lam_window : The wavelength window [in Angstrom] over which to fit, centered on the redshifted line
		plot_results : Plot the results of the line fitting
		"""
		if lines_to_fit==None:
			self.lines_to_fit = {'HEII_1':1640.420,
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
		else:
			self.lines_to_fit = lines_to_fit

		line_sig = lam_window / 2.

		line_results = {}
		for line in self.lines_to_fit:
			zline = self.lines_to_fit[line] * (1.+self.z)
			ifit = np.where((self.lam >= (zline-line_sig)) & (self.lam <= (zline+line_sig)) & (np.isfinite(self.flux)==True))[0]

			if len(ifit)==0:
				print(f'No valid elements for line {line}. Please try a larger or different wavelength window.')
				continue
			# assert len(ifit) > 0, 'No valid elements in spectral window. Please try a larger wavelength window.'

			# Define the spectrum window over which to fit
			xfit, yfit, efit = self.lam[ifit], self.flux[ifit], self.err[ifit]

			# Define the model
			fit_model = models.Gaussian1D(amplitude=np.nanmax(yfit), mean=zline, stddev=60.) + models.Linear1D(slope=0., intercept=0.)
			fit_model.amplitude_0.min = 0.
			fit_model.amplitude_0.max = np.nanmax(yfit)
			fit_model.amplitude_0.fixed = False
			fit_model.stddev_0.min = 0.
			# fit_model.stddev_0.max = 500.
			fit_model.stddev_0.fixed = False
			fit_model.slope_1.fixed = False

			if len(fit_params) > 0:
				for arg in fit_params:
					if 'value' in fit_params[arg]:
						fit_model.__dict__[arg].value = fit_params[arg]['value']
					if 'min' in fit_params[arg]:
						fit_model.__dict__[arg].min = fit_params[arg]['min']
					if 'max' in fit_params[arg]:
						fit_model.__dict__[arg].max = fit_params[arg]['max']
					if 'fixed' in fit_params[arg]:
						fit_model.__dict__[arg].fixed = fit_params[arg]['fixed']


			fit_func = fitting.LevMarLSQFitter(calc_uncertainties=True)
			gauss = fit_func(fit_model, xfit, yfit, weights=1./efit, maxiter=9999999)

			model50 = gauss.copy()
			evaluated50 = gauss(xfit)
			f_gauss = models.Gaussian1D(amplitude=gauss.amplitude_0.value, mean=gauss.mean_0.value, stddev=gauss.stddev_0.value)
			evaluated_gaussian = f_gauss(xfit)
			f_cont = models.Linear1D(slope=gauss.slope_1.value, intercept=gauss.intercept_1.value)
			evaluated_continuum = f_cont(xfit)

			dx = np.diff(xfit)
			dx = np.concatenate([dx,[dx[-1]]])

			# line_flux16 = np.nansum(evaluated16 * normval * dx)
			# line_flux50 = np.nansum(evaluated50 * normval * dx)
			# line_flux84 = np.nansum(evaluated84 * normval * dx)
			# line_flux_err = np.mean([line_flux84-line_flux50,line_flux50-line_flux16])
			# line_flux = ufloat(line_flux50, line_flux_err)

			# line_flux = np.nansum(unp.uarray(evaluated50, efit) * normval * dx)

			line_flux = np.nansum(unp.uarray(evaluated_gaussian, efit) * normval * dx)
			ew = line_flux / (evaluated_continuum[gf.find_nearest(xfit, gauss.mean_0.value)]*normval)
			ew0 = ew / (1.+self.z)

			line_results[line] = {'line_flux':line_flux, 
								  'ew':ew,
								  'ew0':ew0,
								  'gauss':evaluated_gaussian,
								  'continuum':evaluated_continuum,
								  'snr':line_flux.n/line_flux.s,
								  'xfit':xfit, 
								  'yfit':yfit, 
								  'yerr':efit, 
								  'profile50':evaluated50, 
								  'fit_func':gauss}

		self.results = line_results

		if verbose==True:
			for line in self.results:
				print(f'{line} : lam=%0.1f, sig=%0.1f, line_flux=%.3E +/- %.3E (%0.2f), ew=%0.2f +/- %0.2f, ew0=%0.2f +/- %0.2f' % (self.results[line]['fit_func'].mean_0.value, self.results[line]['fit_func'].stddev_0.value,self.results[line]['line_flux'].n,self.results[line]['line_flux'].s,self.results[line]['snr'],self.results[line]['ew'].n,self.results[line]['ew'].s,self.results[line]['ew0'].n,self.results[line]['ew0'].s))

		if plot_results==True:
			fig = plt.figure(figsize=(8.,4.))
			ax = fig.add_subplot(111)
			ax.fill_between(self.lam/10000., self.flux-self.err, self.flux+self.err, alpha=0.2)
			ax.step(self.lam/10000., self.flux, linewidth=1.5, where='mid')
			# ax.set_ylim(ax.set_ylim()[0], np.nanmax(self.flux)*0.25)
			for line in self.results:
				# ylow, yupp = self.results[line]['profile16'], self.results[line]['profile84']
				xfit, yfit = self.results[line]['xfit'], self.results[line]['profile50']
				# ax.fill_between(xfit/10000., ylow, yupp, color='darkred', alpha=0.2)
				ax.step(xfit/10000., yfit, linewidth=2., color='darkred', where='mid')

				ymin = ax.set_ylim()[0] + (ax.set_ylim()[1]-ax.set_ylim()[0]) * 0.6
				ymax = ax.set_ylim()[0] + (ax.set_ylim()[1]-ax.set_ylim()[0]) * 0.7
				zline = self.lines_to_fit[line] * (1.+self.z) / 10000.
				ax.plot([zline,zline], [ymin,ymax], linestyle='-', linewidth=3., color='black')
			gf.style_axes(ax, r'$\lambda_{\rm obs}$ [$\mu$m]', r'$F_{\lambda}$ [cgs]')
			plt.tight_layout()
			plt.show()


	def fit_windows(self, windows=[[1400.,2000],[3500.,4500.],[4700.,5200.]], lam_window=500., normval=1., Niter_err=100, verbose=True, plot_results=False, R=100):
		"""
		lam_window : The wavelength window [in Angstrom] over which to evaluate the line fluxes, centered on the redshifted line
		plot_results : Plot the results of the line fitting
		"""
		if R==100:
			self.lines_to_fit = {'Lya':1215.,
								 'CIV':1549.,
								 'HEII':1640.420,
								 'OIII':1663.4795,
								 'CIII':1908.734,
								 
								 'OII':3728.,
								 'NEIII_1':3869.86,
								 'NEIII_2':3968.59,
								 'HDELTA':4102.8922,
								 # 'HGAMMA':np.mean([4341.6837,4364.436]),
								 'HGAMMA':4341.6837,
								 'OIII_1':4364.436,
								 
								 'HBETA':4862.6830,
								 'OIII_2':4960.295,
								 'OIII_3':5008.240,
								 
								 # 'NII_1':6549.86,
								 'HALPHA':6564.608}
								 # 'NII_2':6585.27}
								 # 'SII':np.mean([6718.295,6732.674])}
		elif R>100:
			self.lines_to_fit = {'Lya':1215.,
								 'CIV':1549.,
								 'HEII':1640.420,
								 'OIII':1663.4795,
								 'CIII':1908.734,
								 
								 'OII':3728.,
								 'NEIII_1':3869.86,
								 'NEIII_2':3968.59,
								 'HDELTA':4102.8922,
								 'HGAMMA':4341.6837,
								 'OIII_1':4364.436,
								 
								 'HBETA':4862.6830,
								 'OIII_2':4960.295,
								 'OIII_3':5008.240,
								 
								 'HALPHA':6564.608,
								 'NII':6585.27,
								 'SII_1':6718.295,
								 'SII_2':6732.674}
	
		
		line_results = {}        
		for nfit,lam_range in enumerate(windows):
			ifit = np.where((self.lam0 > lam_range[0]) & (self.lam0 < lam_range[1]))[0]
	
			if len(ifit) == 0:
				print('No valid elements in spectral window (lam0=%i-%i). Continuing to next window.'%(lam_range[0],lam_range[1]))
				continue
	
			xfit, efit = self.lam[ifit], self.err[ifit]
	
			fit_params, fit_lines = [], []
			Nlines = 0
			for Niter in tqdm.tqdm(range(int(Niter_err)+1)):
				# Perturb the flux to fit by the 1sigma errors of the spectrum
				if Niter==0:
					yfit = self.flux[ifit].copy()
				else:
					yfit = np.random.normal(loc=self.flux[ifit], scale=self.err[ifit])
	
				# Construct the line model
				fit_model = models.Polynomial1D(1)
				for line in self.lines_to_fit:
					zline = self.lines_to_fit[line] * (1.+self.z)
	
					if (zline > min(xfit)) & (zline < max(xfit)):
	
						line_model = models.Gaussian1D(amplitude=np.nanmax(yfit[(xfit > (zline - 100.)) & (xfit < (zline + 100.))]), mean=zline, stddev=50.)
						line_model.amplitude.min = 0.
						line_model.amplitude.max = np.nanmax(yfit[(xfit > (zline - 100.)) & (xfit < (zline + 100.))])
						line_model.amplitude.fixed = False
						line_model.stddev.min = 20.
						line_model.stddev.fixed = False
						line_model.mean.min = zline - 100.
						line_model.mean.max = zline + 100.
		
						fit_model += line_model
						if Niter==0:
							fit_lines.append(line)
							Nlines += 1
	
				fit_func = fitting.LevMarLSQFitter()
				model = fit_func(fit_model, xfit, yfit, weights=1./efit, maxiter=9999999)
				fit_params.append(model)
	
			# Derive and store fitting results and quantities
			lines_results = {}

			cont_model = models.Polynomial1D(1).evaluate(xfit, *[fit_params[0].c0_0.value,fit_params[0].c1_0.value])
			total_spec_profile = cont_model.copy()
			for N in range(len(fit_lines)):
				N += 1
				
				A = fit_params[0].__dict__[f'amplitude_{N}'].value
				mu = fit_params[0].__dict__[f'mean_{N}'].value
				sig = fit_params[0].__dict__[f'stddev_{N}'].value
				
				A_array, mu_array, sig_array = [], [], []
				for Niter in range(len(fit_params)-1):
					Niter += 1
					
					if fit_params[Niter].__dict__[f'amplitude_{N}'].value > 0.:
						A_array.append(fit_params[Niter].__dict__[f'amplitude_{N}'].value)
					if fit_params[Niter].__dict__[f'mean_{N}'].value > 0.:
						mu_array.append(fit_params[Niter].__dict__[f'mean_{N}'].value)
					if fit_params[Niter].__dict__[f'stddev_{N}'].value > 0.:
						sig_array.append(fit_params[Niter].__dict__[f'stddev_{N}'].value)
				
				A_err = np.mean([A - np.percentile(A_array, q=16.), np.percentile(A_array, q=84.)-A])
				mu_err = np.mean([mu - np.percentile(mu_array, q=16.), np.percentile(mu_array, q=84.)-mu])
				sig_err = np.mean([sig - np.percentile(sig_array, q=16.), np.percentile(sig_array, q=84.)-sig])
				
				A = ufloat(A, A_err)
				mu = ufloat(mu, mu_err)
				sig = ufloat(sig, sig_err)

				# Calculate relevant quantities
				gaussian_line = A * unp.exp(-0.5 * (xfit - mu) ** 2 / sig**2)
				total_spec_profile += unp.nominal_values(gaussian_line)

				dx = np.diff(xfit)
				dx = np.concatenate([dx,[dx[-1]]])
				line_flux = np.nansum(gaussian_line * normval * dx)

				yprofile = cont_model + gaussian_line
				ew = abs(np.nansum((1. - yprofile / cont_model) * dx))
				ew0 = ew / (1.+self.z)

				if hasattr(self, 'results'):
					self.results[fit_lines[N-1]] = {'line_flux':line_flux, 
												'snr':line_flux.n/line_flux.s,
												'xfit':xfit,
												'yfit':self.flux[ifit],
												'yerr':efit,
												'full_profile':yprofile,
												'line_profile':gaussian_line,
												'fit_func':fit_params[0],
												'ew':ew,
												'ew0':ew0,
												'params':{'A':A, 'A_array':A_array, 'mu':mu, 'mu_array':mu_array, 'sig':sig, 'sig_array':sig_array},
												}
				else:
					self.results = {fit_lines[N-1]:{'line_flux':line_flux, 
												'snr':line_flux.n/line_flux.s,
												'xfit':xfit,
												'yfit':self.flux[ifit],
												'yerr':efit,
												'full_profile':yprofile,
												'line_profile':gaussian_line,
												'fit_func':fit_params[0],
												'ew':ew,
												'ew0':ew0,
												'params':{'A':A, 'A_array':A_array, 'mu':mu, 'mu_array':mu_array, 'sig':sig, 'sig_array':sig_array},
												}}


			if hasattr(self, 'results'):
				self.results[nfit] = {'xfit':xfit, 'yfit':yfit, 'yerr':efit, 'profile':total_spec_profile, 'fit_lines':fit_lines}
			else:
				self.results = {nfit:{'xfit':xfit, 'yfit':yfit, 'yerr':efit, 'profile':total_spec_profile, 'fit_lines':fit_lines}}

		# self.results = line_results

		if verbose==True:
			for line in self.results:
				if line in [0,1,2,3,4,5,6,7,8,9,10,'fit_lines']:
					continue
				print(f'{line} : A=%.3E\u00B1%.3E, mu=%0.1f\u00B1%0.1f, sig=%0.1f\u00B1%0.1f, line_flux=%.3E\u00B1%.3E (SNR=%0.2f)' % (self.results[line]['params']['A'].n/normval,self.results[line]['params']['A'].s/normval,self.results[line]['params']['mu'].n,self.results[line]['params']['mu'].s, self.results[line]['params']['sig'].n,self.results[line]['params']['sig'].s, self.results[line]['line_flux'].n/normval, self.results[line]['line_flux'].s/normval, self.results[line]['snr']))
				# print(f'{line} : A=%.3E\u00B1%.3E, mu=%0.1f\u00B1%0.1f, sig=%0.1f\u00B1%0.1f, line_flux=%.3E\u00B1%.3E' % (self.results[line]['params']['A'].n/normval,self.results[line]['params']['A'].s/normval,self.results[line]['params']['mu'].n,self.results[line]['params']['mu'].s, self.results[line]['params']['sig'].n,self.results[line]['params']['sig'].s, self.results[line]['line_flux'].n/normval, self.results[line]['line_flux'].s/normval))

		if plot_results==True:
			fig = plt.figure(figsize=(8.,4.))
			ax = fig.add_subplot(111)
			# ax.fill_between(self.lam/10000., self.flux-self.err, self.flux+self.err, alpha=0.2)
			ax.step(self.lam, self.flux, linewidth=1.5, where='mid')
			maxval = abs(np.nanmax(self.flux[(self.lam > 3.)]))
			ax.set_ylim(-maxval*0.25, maxval*0.75)
			for window in range(len(windows)):
				xfit, yfit = self.results[window]['xfit'], self.results[window]['profile']
				ax.plot(xfit, yfit, linewidth=2., color='darkred')

				ymin = ax.set_ylim()[0] + (ax.set_ylim()[1]-ax.set_ylim()[0]) * 0.6
				ymax = ax.set_ylim()[0] + (ax.set_ylim()[1]-ax.set_ylim()[0]) * 0.7
				for line in self.results[window]['fit_lines']:
					zline = self.lines_to_fit[line] * (1.+self.z)
					ax.plot([zline,zline], [ymin,ymax], linestyle='-', linewidth=3., color='black')
			ax.plot(ax.set_xlim(), [0.,0.], linestyle='--', linewidth=1., color='black')
			gf.style_axes(ax, r'$\lambda_{\rm obs}$ [$\mu$m]', r'$F_{\lambda}$ [cgs]')
			plt.tight_layout()
			plt.show()


	def fit_lya_skew(self, lam_window=2500., normval=1., Niter_err=100, verbose=True, plot_results=False, R=100):

		# Skew-normal function to fit Lya
		class SkewedGaussian1D(Fittable1DModel):
			amplitude = Parameter()
			mean = Parameter()
			stddev = Parameter()
			skew = Parameter()
		
			@staticmethod
			def evaluate(x, amplitude, mean, stddev, skew):
				norm_x = (x - mean) / stddev
				return amplitude * np.exp(-0.5 * norm_x**2) * (1 + skew * norm_x)


		zline = 1215.670 * (1.+self.z)
		ifit = np.where((self.lam > zline-(lam_window/2.)) & (self.lam < zline+(lam_window/2.)))[0]

		assert len(ifit)>2, 'Need at least 3 spectral elements to fit the line. There are less than 3.'
	
		xfit, efit = self.lam[ifit], self.err[ifit]

		fit_params = []
		for Niter in tqdm.tqdm(range(int(Niter_err)+1)):
			# Perturb the flux to fit by the 1sigma errors of the spectrum
			if Niter==0:
				yfit = self.flux[ifit].copy()
			else:
				yfit = np.random.normal(loc=self.flux[ifit], scale=self.err[ifit])

			# Initial model guess
			ibad = np.where(np.isfinite(yfit)==False)[0]
			yfit[ibad] = 0.
			efit[ibad] = 1.e20

			init_poly = models.Polynomial1D(1)
			init_skew_gauss = SkewedGaussian1D(amplitude=np.nanmax(yfit[(xfit > (zline - 100.)) & (xfit < (zline + 100.))]),
							   mean=zline,
							   stddev=50.,
							   skew=0.1)
			skewed_gauss = init_poly + init_skew_gauss
			
			skewed_gauss.amplitude_1.min = 0.
			skewed_gauss.amplitude_1.max = np.nanmax(yfit[(xfit > (zline - 100.)) & (xfit < (zline + 100.))])
			skewed_gauss.amplitude_1.value = np.nanmax(yfit[(xfit > (zline - 100.)) & (xfit < (zline + 100.))])
			skewed_gauss.amplitude_1.fixed = False
			skewed_gauss.stddev_1.min = 0.
			skewed_gauss.stddev_1.fixed = False
			skewed_gauss.mean_1.min = zline - 25.
			skewed_gauss.mean_1.max = zline + 25.
			skewed_gauss.skew_1.min = 0.
			skewed_gauss.skew_1.max = 0.5

			fit_func = fitting.LevMarLSQFitter()
			model = fit_func(skewed_gauss, xfit, yfit, weights=1./efit, maxiter=9999999)
			fit_params.append(model)


		# Evaluate the best-fit model
		cont_model = models.Polynomial1D(1).evaluate(xfit, *[fit_params[0].c0_0.value,fit_params[0].c1_0.value])
		total_spec_profile = cont_model.copy()

		A = fit_params[0].__dict__[f'amplitude_1'].value
		mu = fit_params[0].__dict__[f'mean_1'].value
		sig = fit_params[0].__dict__[f'stddev_1'].value
		skew = fit_params[0].__dict__[f'skew_1'].value

		A_array, mu_array, sig_array, skew_array = [], [], [], []
		for Niter in range(len(fit_params)-1):
			Niter += 1
			
			if fit_params[Niter].__dict__[f'amplitude_1'].value > 0.:
				A_array.append(fit_params[Niter].__dict__[f'amplitude_1'].value)
			if fit_params[Niter].__dict__[f'mean_1'].value > 0.:
				mu_array.append(fit_params[Niter].__dict__[f'mean_1'].value)
			if fit_params[Niter].__dict__[f'stddev_1'].value > 0.:
				sig_array.append(fit_params[Niter].__dict__[f'stddev_1'].value)
			if fit_params[Niter].__dict__[f'skew_1'].value > 0.:
				skew_array.append(fit_params[Niter].__dict__[f'skew_1'].value)
		
		A_err = np.mean([A - np.percentile(A_array, q=16.), np.percentile(A_array, q=84.)-A])
		mu_err = np.mean([mu - np.percentile(mu_array, q=16.), np.percentile(mu_array, q=84.)-mu])
		sig_err = np.mean([sig - np.percentile(sig_array, q=16.), np.percentile(sig_array, q=84.)-sig])
		try:
			skew_err = np.mean([skew - np.percentile(skew_array, q=16.), np.percentile(skew_array, q=84.)-skew])
		except:
			skew_err = np.std(skew_array)
		
		A = ufloat(A, A_err)
		mu = ufloat(mu, mu_err)
		sig = ufloat(sig, sig_err)
		skew = ufloat(skew, skew_err)

		# Calculate relevant quantities
		norm_x = (xfit - mu) / sig
		lya_line = A * unp.exp(-0.5 * norm_x**2) * (1 + skew * norm_x)
		total_spec_profile += unp.nominal_values(lya_line)

		dx = np.diff(xfit)
		dx = np.concatenate([dx,[dx[-1]]])
		line_flux = np.nansum(lya_line * normval * dx)

		yprofile = cont_model + lya_line
		ew = abs(np.nansum((1. - yprofile / cont_model) * dx))
		ew0 = ew / (1.+self.z)

		if hasattr(self, 'results'):
			self.results['LYA'] = {'line_flux':line_flux, 
										'snr':line_flux.n/line_flux.s,
										'xfit':xfit,
										'yfit':self.flux[ifit],
										'yerr':efit,
										'full_profile':yprofile,
										'line_profile':lya_line,
										'fit_func':fit_params[0],
										'ew':ew,
										'ew0':ew0,
										'params':{'A':A, 'A_array':A_array, 'mu':mu, 'skew':skew, 'mu_array':mu_array, 'sig':sig, 'sig_array':sig_array, 'skew_array':skew_array},
										}
		else:
			self.results = {'LYA':{'line_flux':line_flux, 
										'snr':line_flux.n/line_flux.s,
										'xfit':xfit,
										'yfit':self.flux[ifit],
										'yerr':efit,
										'full_profile':yprofile,
										'line_profile':lya_line,
										'fit_func':fit_params[0],
										'ew':ew,
										'ew0':ew0,
										'params':{'A':A, 'A_array':A_array, 'mu':mu, 'skew':skew, 'mu_array':mu_array, 'sig':sig, 'sig_array':sig_array, 'skew_array':skew_array},
										}}

		if verbose==True:
			print(f'LYA : A=%.3E\u00B1%.3E, mu=%0.1f\u00B1%0.1f, sig=%0.1f\u00B1%0.1f, skew=%0.1f\u00B1%0.1f, line_flux=%.3E\u00B1%.3E (SNR=%0.2f)' % (self.results['LYA']['params']['A'].n/normval,self.results['LYA']['params']['A'].s/normval,self.results['LYA']['params']['mu'].n,self.results['LYA']['params']['mu'].s, self.results['LYA']['params']['sig'].n,self.results['LYA']['params']['sig'].s, self.results['LYA']['params']['skew'].n,self.results['LYA']['params']['skew'].s, self.results['LYA']['line_flux'].n/normval, self.results['LYA']['line_flux'].s/normval, self.results['LYA']['snr']))

		if plot_results==True:
			fig = plt.figure(figsize=(8.,4.))
			ax = fig.add_subplot(111)
			ax.step(self.lam, self.flux, linewidth=1.5, where='mid')
			ax.plot(self.lam, self.err, linestyle=':', color='grey')
			maxval = abs(np.nanmax(self.flux[(self.lam > 3.)]))
			ax.set_ylim(-maxval*0.25, maxval*0.75)
			xfit, yfit = self.results['LYA']['xfit'], unp.nominal_values(self.results['LYA']['full_profile'])
			ax.plot(xfit, yfit, linewidth=2., color='darkred')

			ymin = ax.set_ylim()[0] + (ax.set_ylim()[1]-ax.set_ylim()[0]) * 0.6
			ymax = ax.set_ylim()[0] + (ax.set_ylim()[1]-ax.set_ylim()[0]) * 0.7
			zline = 1215.670 * (1.+self.z)
			ax.plot([zline,zline], [ymin,ymax], linestyle='-', linewidth=3., color='black')
			ax.plot(ax.set_xlim(), [0.,0.], linestyle='--', linewidth=1., color='black')
			gf.style_axes(ax, r'$\lambda_{\rm obs}$ [$\mu$m]', r'$F_{\lambda}$ [cgs]')
			plt.tight_layout()
			plt.show()


	def fit_lya_gauss(self, lam_window=2500., normval=1., Niter_err=100, verbose=True, plot_results=False, R=100):

		zline = 1215.670 * (1.+self.z)
		ifit = np.where((self.lam > zline-(lam_window/2.)) & (self.lam < zline+(lam_window/2.)))[0]

		assert len(ifit)>2, 'Need at least 3 spectral elements to fit the line. There are less than 3.'
	
		xfit, efit = self.lam[ifit], self.err[ifit]

		fit_params = []
		for Niter in tqdm.tqdm(range(int(Niter_err)+1)):
			# Perturb the flux to fit by the 1sigma errors of the spectrum
			if Niter==0:
				yfit = self.flux[ifit].copy()
			else:
				yfit = np.random.normal(loc=self.flux[ifit], scale=self.err[ifit])

			# Initial model guess
			ibad = np.where(np.isfinite(yfit)==False)[0]
			yfit[ibad] = 0.
			efit[ibad] = 1.e20

			init_poly = models.Polynomial1D(1)
			init_gauss = models.Gaussian1D(amplitude=np.nanmax(yfit[(xfit > (zline - 100.)) & (xfit < (zline + 100.))]), mean=zline, stddev=50.)
			init_gauss.amplitude.min = 0.
			init_gauss.amplitude.max = np.nanmax(yfit[(xfit > (zline - 100.)) & (xfit < (zline + 100.))])
			init_gauss.amplitude.fixed = False
			init_gauss.stddev.min = 20.
			init_gauss.stddev.fixed = False
			init_gauss.mean.min = zline - 500.
			init_gauss.mean.max = zline + 500.
		
			profile_to_fit = init_poly + init_gauss
			# profile_to_fit.c0_0.value = 0.
			# profile_to_fit.c0_0.fixed = True
			# profile_to_fit.c1_0.value = 0.
			# profile_to_fit.c1_0.fixed = True
			fit_func = fitting.LevMarLSQFitter()
			model = fit_func(profile_to_fit, xfit, yfit, weights=1./efit, maxiter=9999999)
			fit_params.append(model)


		# Evaluate the best-fit model
		cont_model = models.Polynomial1D(1).evaluate(xfit, *[fit_params[0].c0_0.value,fit_params[0].c1_0.value])
		total_spec_profile = cont_model.copy()

		A = fit_params[0].__dict__[f'amplitude_1'].value
		mu = fit_params[0].__dict__[f'mean_1'].value
		sig = fit_params[0].__dict__[f'stddev_1'].value

		A_array, mu_array, sig_array = [], [], []
		for Niter in range(len(fit_params)-1):
			Niter += 1
			
			if fit_params[Niter].__dict__[f'amplitude_1'].value > 0.:
				A_array.append(fit_params[Niter].__dict__[f'amplitude_1'].value)
			if fit_params[Niter].__dict__[f'mean_1'].value > 0.:
				mu_array.append(fit_params[Niter].__dict__[f'mean_1'].value)
			if fit_params[Niter].__dict__[f'stddev_1'].value > 0.:
				sig_array.append(fit_params[Niter].__dict__[f'stddev_1'].value)
		
		A_err = np.mean([A - np.percentile(A_array, q=16.), np.percentile(A_array, q=84.)-A])
		mu_err = np.mean([mu - np.percentile(mu_array, q=16.), np.percentile(mu_array, q=84.)-mu])
		sig_err = np.mean([sig - np.percentile(sig_array, q=16.), np.percentile(sig_array, q=84.)-sig])
		
		A = ufloat(A, A_err)
		mu = ufloat(mu, mu_err)
		sig = ufloat(sig, sig_err)

		# Calculate relevant quantities
		lya_line = A * unp.exp(-0.5 * (xfit - mu) ** 2 / sig**2)
		total_spec_profile += unp.nominal_values(lya_line)

		dx = np.diff(xfit)
		dx = np.concatenate([dx,[dx[-1]]])
		line_flux = np.nansum(lya_line * normval * dx)

		yprofile = cont_model + lya_line
		try:
			ew = abs(np.nansum((1. - yprofile / cont_model) * dx))
			ew0 = ew / (1.+self.z)
		except:
			ew = -99.
			ew0 = -99.

		if hasattr(self, 'results'):
			self.results['LYA'] = {'line_flux':line_flux, 
										'snr':line_flux.n/line_flux.s,
										'xfit':xfit,
										'yfit':self.flux[ifit],
										'yerr':efit,
										'full_profile':yprofile,
										'line_profile':lya_line,
										'fit_func':fit_params[0],
										'ew':ew,
										'ew0':ew0,
										'params':{'A':A, 'A_array':A_array, 'mu':mu, 'mu_array':mu_array, 'sig':sig, 'sig_array':sig_array},
										}
		else:
			self.results = {'LYA':{'line_flux':line_flux, 
										'snr':line_flux.n/line_flux.s,
										'xfit':xfit,
										'yfit':self.flux[ifit],
										'yerr':efit,
										'full_profile':yprofile,
										'line_profile':lya_line,
										'fit_func':fit_params[0],
										'ew':ew,
										'ew0':ew0,
										'params':{'A':A, 'A_array':A_array, 'mu':mu, 'mu_array':mu_array, 'sig':sig, 'sig_array':sig_array},
										}}

		if verbose==True:
			print(f'LYA : A=%.3E\u00B1%.3E, mu=%0.1f\u00B1%0.1f, sig=%0.1f\u00B1%0.1f, line_flux=%.3E\u00B1%.3E (SNR=%0.2f)' % (self.results['LYA']['params']['A'].n/normval,self.results['LYA']['params']['A'].s/normval,self.results['LYA']['params']['mu'].n,self.results['LYA']['params']['mu'].s, self.results['LYA']['params']['sig'].n,self.results['LYA']['params']['sig'].s, self.results['LYA']['line_flux'].n/normval, self.results['LYA']['line_flux'].s/normval, self.results['LYA']['snr']))

		if plot_results==True:
			fig = plt.figure(figsize=(8.,4.))
			ax = fig.add_subplot(111)
			ax.step(self.lam, self.flux, linewidth=1.5, where='mid')
			ax.plot(self.lam, self.err, linestyle=':', color='grey')
			maxval = abs(np.nanmax(self.flux[(self.lam < 20000.)]))
			ax.set_ylim(-maxval*0.25, maxval*1.25)
			xfit, yfit = self.results['LYA']['xfit'], unp.nominal_values(self.results['LYA']['full_profile'])
			ax.plot(xfit, yfit, linewidth=2., color='darkred')

			ymin = ax.set_ylim()[0] + (ax.set_ylim()[1]-ax.set_ylim()[0]) * 0.6
			ymax = ax.set_ylim()[0] + (ax.set_ylim()[1]-ax.set_ylim()[0]) * 0.7
			zline = 1215.670 * (1.+self.z)
			ax.plot([zline,zline], [ymin,ymax], linestyle='-', linewidth=3., color='black')
			ax.plot(ax.set_xlim(), [0.,0.], linestyle='--', linewidth=1., color='black')
			gf.style_axes(ax, r'$\lambda_{\rm obs}$ [$\mu$m]', r'$F_{\lambda}$ [cgs]')
			plt.tight_layout()
			plt.show()


	def fit_upper_limit(self, clam=None, npix=3, plot_results=False):
		from astropy.constants import c
		import astropy.units as u
		c = c.to(u.km/u.s).value
		
		assert clam!=None, 'Need to provide a central wavelength over which to calculate the upper limit.'
		
		cent = gf.find_nearest(clam, self.lam)
		signal_limits = [int(cent-(npix/2)),int(cent+(npix/2)+1)]
	
		H = np.nanstd(self.flux[signal_limits[0]:signal_limits[-1]])
		dlam = self.lam[signal_limits[-1]] - self.lam[signal_limits[0]]
		dv = (dlam / clam) * c
		line_flux = H * dlam / (2.35 * 0.3989)
		
		cont = np.nanmedian(self.flux[signal_limits[0]:signal_limits[-1]])
		ew = line_flux / cont
		ew0 = ew / (1.+self.z)
		print(clam/(1.+self.z), line_flux, cont, ew, ew0)

		results = {'line_flux':line_flux, 'cont':cont, 'ew':ew, 'ew0':ew0}
		
		if plot_results==True:
			fig = plt.figure(figsize=(8.,4.))
			ax = fig.add_subplot(111)
			ax.step(self.lam0, self.flux)
			ax.fill_between([self.lam0[signal_limits[0]],self.lam0[signal_limits[1]]], [ax.set_ylim()[0],ax.set_ylim()[0]], [ax.set_ylim()[1],ax.set_ylim()[1]], color='cyan', alpha=0.3)
			gf.style_axes(ax, r'$\lambda_{\rm obs}$ [$\mu$m]', r'$F_{\lambda}$ [cgs]')
			plt.tight_layout()
			plt.show()
		
		return results

def inverse_variance_mean(flux_array, error_array):
	"""
	Combine multiple JWST 1D spectra using inverse-variance weighting.

	Parameters:
		flux_array: np.ndarray
			2D array of shape (N, n) containing flux values for N spectra.
		error_array: np.ndarray
			2D array of shape (N, n) containing associated uncertainties.

	Returns:
		combined_wavelength, combined_flux, combined_error: np.ndarray
			Arrays representing the combined spectrum.
	"""
	if type(flux_array) == list:
		flux_array = np.array(flux_array)
	if type(error_array) == list:
		error_array = np.array(error_array)
	
	if flux_array.shape != error_array.shape:
		raise ValueError("Flux and error arrays must have the same shape.")

	# Inverse variance weights
	weights = 1. / error_array**2

	# Weighted mean calculation
	combined_flux = np.nansum(flux_array * weights, axis=0) / np.nansum(weights, axis=0)
	combined_error = np.sqrt(1. / np.nansum(weights, axis=0))

	return combined_flux, combined_error


def correct_dust_extinction(lam, flux, ratio, main_line='ha', rv=2.74, dust_law='fm07'):
	"""
	lam : observed-frame wavelength array in Angstroms
	flux : spectral flux array (probably any units but to be safe do this in erg/s/cm2/A)
	ratio : Ratio of a Balmer line (e.g., Halpha, Hgamma, Hdelta) to Hbeta
	main_line : the Balmer line used for the ratio wit Hbeta
	"""
	if dust_law in ['Cardelli','cardelli','c89','ccm89']:
		k_ha = 2.534
		k_hb = 3.608
		k_hg = 4.172
		k_hd = 4.438
	elif dust_law in ['Fitzpatrick','fitzpatrick','f07','fm07']:
		k_ha = 1.4423
		k_hb = 1.0877
		k_hg = 0.9254
		k_hd = 0.8719
		
	# ha_ratio = 2.790
	# hg_ratio = 0.473
	# hd_ratio = 0.263
	ha_ratio = 2.86
	hg_ratio = 0.473
	hd_ratio = 0.263

	if main_line in ['ha','halpha','HA','HALPHA']:
		ebv = (2.5 / (k_hb-k_ha)) * unp.log10(ratio / ha_ratio)
	elif main_line in ['hg','hgamma','HG','HGAMMA','hd','hdelta','HD','HDELTA']:
		ebv = (-2.5 / (k_hb-k_hg)) * unp.log10(hg_ratio / ratio)

	if ebv <= 0.:
		print('Negative excess, assuming zero dust')
		return flux
	else:
		if dust_law in ['Cardelli','cardelli','c89','ccm89']:
			Av = ebv * rv
			Alam = extinction.ccm89(lam, a_v=Av, r_v=rv)
		elif dust_law in ['Fitzpatrick','fitzpatrick','f07','fm07']:
			Av = ebv * 3.1
			Alam = extinction.fm07(lam, a_v=Av)
		flux_corr = flux * 10**(0.4 * Alam)
		return flux_corr



def emcee_fit(fit_dict, plot_corner=False):
	"""
	In fit_dict make sure to include:
		xvals : the x-array to fit over
		yvals : the y-array to fit over
		yerr : the 1sig array corresponding to yvals
		log_likelihood : the lo
	"""

	from scipy.optimize import minimize
	import emcee
	import corner
	import numpy as np
	import warnings
	warnings.filterwarnings("ignore")
	
	# Set default parameters
	assert ('xvals' in fit_dict) & ('yvals' in fit_dict) & ('yerr' in fit_dict), 'Need to provide wavelength, flux and uncertainties arrays to fit...'
	
	if 'emcee_iter' not in fit_dict:
		fit_dict['emcee_iter'] = 500
	log_likelihood = fit_dict['log_likelihood']
	log_prior = fit_dict['log_prior']
	
	xvals, yvals, yerr = fit_dict['xvals'], fit_dict['yvals'], fit_dict['yerr']
	
	if log_prior!=None:
		def log_probability(theta, xvals, yvals, yerr):
			lp = log_prior(theta)
			loglike = log_likelihood(theta, xvals, yvals, yerr)
			if (np.isfinite(lp)==False) | (np.isfinite(loglike)==False):
				return -np.inf
			return lp + loglike
	else:
		def log_probability(theta, xvals, yvals, yerr):
			loglike = log_likelihood(theta, xvals, yvals, yerr)
			if (np.isfinite(loglike)==False):
				return -np.inf
			return loglike
		
	nll = lambda *args: -log_likelihood(*args)
	soln = minimize(nll, fit_dict['p0'], args=(xvals, yvals, yerr))
	pos = soln.x + 1e-4 * np.random.randn(fit_dict['Nparams']*25, fit_dict['Nparams'])
	nwalkers, ndim = pos.shape
	
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xvals, yvals, yerr))
	sampler.run_mcmc(pos, fit_dict['emcee_iter'], progress=True)
	
	flat_samples = sampler.get_chain(discard=50, thin=15, flat=True)
	params = []
	params_err = []
	for i in range(ndim):
		mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
		q = np.diff(mcmc)
		params.append(mcmc[1])
		params_err.append(np.mean(q))
	
	print('Best fit parameters :')
	for i in range(len(params)):
		print(f'Parameter {i} : %0.2f+/-%0.2f' % (params[i],params_err[i]))
		
	if plot_corner==True:
		fig = corner.corner(flat_samples)
	return params, params_err


def stack_with_nircam(spectra, names, z_stack=None, method='median', scale_filt='JWST_NIRCam.F200W', apply_slitloss=True, clip=True, filename=None, save=['full','CIII','Balmer','OIIIHbeta','HeI','Halpha'], plot=True, ylims=None):

	tab = ascii.read('/Users/guidorb/Dropbox/Postdoc/highz_galaxies_combined.dat')

	assert filename is not None, "Need to specify file name including 'z[]p[]' keyword!"
	
	output_spec = {}
	output_photo = {}
	
	### Stack NIRCam photometry for slit loss verification later ###
	for name in names:
		i = np.where(tab['name'] == name)[0]
		mu = tab['mu'][i]

		if 'photo' not in spectra[name]:
			continue
		photo = spectra[name]['photo']
		
		for nfilt, filt in enumerate(photo['filters']):
			filt = filt.split('/')[-1]
			if filt not in ['JWST_NIRCAM.F090W','JWST_NIRCAM.F115W','JWST_NIRCAM.F150W','JWST_NIRCAM.F200W','JWST_NIRCAM.F277W','JWST_NIRCAM.F356W','JWST_NIRCAM.F444W']:
				continue
			
			# Add filter to catalog if its not there
			if filt not in output_photo:
				output_photo[filt] = {'lam_cent':photo['lam_cent'][nfilt], 'lam_err':photo['lam_err'][nfilt], 'iflux_nircam':[]}
			output_photo[filt]['iflux_nircam'].append(photo['flux_aper'][nfilt]*1000. / mu) # fluxes in nJy and corrected for magnification
	
	for filt in output_photo.keys():
		# apply sigma clip
		output_photo[filt]['iflux_nircam'] = sigma_clip(output_photo[filt]['iflux_nircam'], sigma=3., masked=False)

		percentiles = np.percentile(output_photo[filt]['iflux_nircam'], [16.,84.])
		med = np.nanmedian(output_photo[filt]['iflux_nircam'])
		
		output_photo[filt]['istd_nircam'] = np.mean([med-percentiles[0], percentiles[1]-med]) / np.sqrt(len(output_photo[filt]['iflux_nircam']))
		output_photo[filt]['iflux_nircam'] = med

	### Stack 1D spectrum ###
	lam0_min, lam0_max = 0., 5.3
	ilam = np.arange(lam0_min*10000., lam0_max*10000.,10.)
	
	output_spec['ilam'] = ilam

	### Stack 1D and 2D spectrum ###
	stack_flux = []
	stack_flux_2d = []
	# xvals = ilam / (1.+z_stack)
	# yvals = np.arange(0,31,1)
	for name in names:
		# 1D 
		i = np.where(tab['name'] == name)[0]
		z_spec = tab['z_spec'][i]

		lam, spec, _, mask = af.get_prism_spectrum(spectra, name, clip=False)
		spec = spec * 1000. # convert to nJy
		
		lam0 = lam / (1.+z_spec)
		fl = np.nanmedian(spec[(lam0 > 0.15) & (lam0 < 0.2)])
		spec = spec / fl # normalize to unity

		spec[(mask == 1)] = 0.
		
		lam0 = lam0 * 10000.
		if z_stack > 0.:
			lam0 = lam0 * (1.+z_stack)
		
		ispec = np.interp(ilam, lam0, spec)
		stack_flux.append(ispec)

		# # 2D
		# img = spectra[name]['prism']['2d']
		# if 'SRCYPIX' not in spectra[name]['prism']:
		# 	continue
		# ycent = int(spectra[name]['prism']['SRCYPIX'])
		# idx_norm = np.where((lam0 > 1500.) & (lam0 < 2000.))[0]
		# norm = np.nanmedian(img[ycent][idx_norm])
		# img = img / norm
		# img[(np.isfinite(img)==False)] = 0.
		
		# # This set of code should be removed once the catalog is updated. Currently handles exception where the pixel grid is not uniform across the sample
		# if np.shape(img)!=(31, 414):
		# 	f = scipy.interpolate.interp2d(np.arange(np.shape(img)[1]), np.arange(np.shape(img)[0]), img, fill_value=0., kind='cubic')
		# 	img = f(np.arange(414), np.arange(31))
		
		# f = scipy.interpolate.interp2d(lam0, yvals, img, fill_value=0., kind='cubic')
		# img0 = f(xvals, yvals)
		# img0[(img0 == 0.)] = np.nan
		# stack_flux_2d.append(img0)

	stack_flux = np.array(stack_flux)
	# stack_flux_2d = np.array(stack_flux_2d)
	
	ilam0 = ilam / (1.+z_stack)
	output_spec['ilam0'] = ilam0
	
	iflux = []
	istd = []
	iscatter = []
	iNgals = []
	for i in range(len(stack_flux.T)):
		if clip==False:
			fluxes = stack_flux.T[i][(stack_flux.T[i] != 0.) & (np.isfinite(stack_flux.T[i])==True)]
			n = len(fluxes)
			iNgals.append(n)
			if n==0:
				iflux.append(0.)
				istd.append(0.)
				iscatter.append(0.)
				continue
			
			if method=='median':
				iflux.append(np.median(fluxes))
			elif method=='mean':
				iflux.append(np.mean(fluxes))
			iscatter.append(np.mean([(iflux[-1]-np.percentile(fluxes,16)),(np.percentile(fluxes,84)-iflux[-1])]))
			istd.append(np.mean([(iflux[-1]-np.percentile(fluxes,16)),(np.percentile(fluxes,84)-iflux[-1])])/np.sqrt(n))
				
		elif clip==True:
			flux_arr, minval, maxval, ival = af.custom_sigma_clip(stack_flux.T[i][(stack_flux.T[i] != 0.) & (np.isfinite(stack_flux.T[i])==True)], low=3., high=3., op=method)
			n = len(flux_arr)
			iNgals.append(n)
			if n==0:
				iflux.append(0.)
				istd.append(0.)
				iscatter.append(0.)
				continue
			
			if method=='median':
				iflux.append(np.median(flux_arr))
			elif method=='mean':
				iflux.append(np.mean(flux_arr))
	
			iscatter.append(np.mean([(iflux[-1]-np.percentile(flux_arr,16)),(np.percentile(flux_arr,84)-iflux[-1])]))
			istd.append(np.mean([(iflux[-1]-np.percentile(flux_arr,16)),(np.percentile(flux_arr,84)-iflux[-1])])/np.sqrt(n))
				
	iflux = np.array(iflux)
	istd = np.array(istd)
	iscatter = np.array(iscatter)
	iNgals = np.array(iNgals)
	
	# Save spectral and photometric fluxes prior to any scaling or slit loss corrections
	output_spec['iflux_nocorr'] = iflux.copy()
	output_spec['istd_nocorr'] = istd.copy()
	output_spec['iscatter_nocorr'] = iscatter.copy()
	output_spec['ngals'] = iNgals.copy()
	# output_spec['2d'] = np.nanmedian(stack_flux_2d, axis=0)

	for filt in output_photo.keys():
		ph = gf.photo_from_filter(output_spec['ilam']/10000., output_spec['iflux_nocorr'], filt=filt)
		output_photo[filt]['iflux_nocorr'] = ph
		
	### Slit loss corrections ###
	# Scale the fluxes to a filter of choice
	if scale_filt!=None:
		scale_filt = scale_filt.upper()
		ph = gf.photo_from_filter(output_spec['ilam']/10000., output_spec['iflux_nocorr'], filt=scale_filt)
		nircam = output_photo[scale_filt]['iflux_nircam']
		factor = nircam / ph
		
		output_spec['iflux_scaled'] = output_spec['iflux_nocorr'] * factor
		output_spec['istd_scaled'] = output_spec['istd_nocorr'] * factor
		output_spec['iscatter_scaled'] = output_spec['iscatter_nocorr'] * factor
		# output_spec['2d'] = output_spec['2d'] * factor
		
		# Get pseudo-photometry of scaled spectrum prior to wavelength slit loss correction
		for filt in output_photo.keys():
			ph = gf.photo_from_filter(output_spec['ilam']/10000., output_spec['iflux_scaled'], filt=filt)
			output_photo[filt]['iflux_scaled'] = ph
		
	# Correct for wavelength-dependent slit loss
	if apply_slitloss==True:

		def fit_slitloss(x, A, B, C):
			y = (A*x**2) + (B*x**1) + C
			return y

		slitloss_lam, slitloss_factor = [], []
		ref_lam, _, _ = gf.get_filter_info(scale_filt)
		ref_lam = ref_lam * 10000.
		for filt in output_photo.keys():
			lam_cent = output_photo[filt]['lam_cent']*10000.
			lam_err = output_photo[filt]['lam_err']*10000.

			nircam = ufloat(output_photo[filt]['iflux_nircam'], output_photo[filt]['istd_nircam'])
			nirspec = ufloat(output_photo[filt]['iflux_scaled'], 0.)
			loss_fact = nircam / nirspec

			# if ((lam_cent+lam_err) < (1300. * (1+z_stack))) | (loss_fact < 1.):
			# if (lam_cent < ref_lam):
			if (lam_cent < (1215.*(1.+z_stack))):
				continue
			slitloss_lam.append(lam_cent)
			slitloss_factor.append(loss_fact)
			
		slitloss_lam = np.array(slitloss_lam)
		slitloss_factor = np.array(slitloss_factor)
		popt, pcov = curve_fit(fit_slitloss, slitloss_lam, unp.nominal_values(slitloss_factor), sigma=unp.std_devs(slitloss_factor), absolute_sigma=True, bounds=[(0.,0.,0.),(1.,1.,1.)])
		fit = np.polyval(popt, output_spec['ilam'])
		fit[(fit < 1.)] = 1.

		output_spec['slitloss_corr'] = fit
		
		fig = plt.figure(figsize=(8.,4.5))
		ax = fig.add_subplot(111)
		ax.set_xlim(0.6,5.3)
		ax.set_ylim(0.,2.)
		ax.errorbar(slitloss_lam/10000., unp.nominal_values(slitloss_factor), yerr=unp.std_devs(slitloss_factor), marker='s', color='C0', markeredgecolor='navy', markersize=10., linestyle='')
		ax.plot(output_spec['ilam']/10000., fit, color='darkred')
		gf.style_axes(ax, r'Observed Wavelength [$\mu$m]', 'Slit loss factor [NIRcam / NIRSpec]')
		plt.tight_layout()
		plt.show()
		
		output_spec['iflux_corr'] = output_spec['iflux_scaled'] * fit
		output_spec['istd_corr'] = output_spec['istd_scaled'] * fit
		output_spec['iscatter_corr'] = output_spec['iscatter_scaled'] * fit
		# for i in range(len(output_spec['2d'].T)):
		# 	output_spec['2d'].T[i] = output_spec['2d'].T[i] * fit
		
		# Get pseudo-photometry of scaled and slit loss corrected spectrum
		for filt in output_photo.keys():
			ph = gf.photo_from_filter(output_spec['ilam']/10000., output_spec['iflux_corr'], filt=filt)
			output_photo[filt]['iflux_corr'] = ph
		
	# Generate uncertainty mask for emission lines
	imask = af.generate_line_mask(output_spec['ilam'], z_spec=z_stack, dv=4000)
	output_spec['istd_corr_maskedlines'] = output_spec['istd_corr'].copy()
	output_spec['istd_corr_maskedlines'][(imask == 1)] = 1.e20
	
	for ext in save:
		if ext == 'full':
			min_lam0, max_lam0 = 0., 20000.
			
			save_dict = {'spec':output_spec, 'phot':output_photo}
			pickle.dump(save_dict, open(f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/input_stacks/{filename}.p', 'wb'))
			
		elif ext == 'CIII':
			min_lam0, max_lam0 = 1400., 2500.
		elif ext == 'Balmer':
#             min_lam0, max_lam0 = 3375., 4605.
			min_lam0, max_lam0 = 3200., 4800.
		elif ext == 'OIIIHbeta':
#             min_lam0, max_lam0 = 4605., 5325.
			min_lam0, max_lam0 = 4550., 5325.
		elif ext == 'HeI':
			min_lam0, max_lam0 = 5750., 6000.
		elif ext == 'Halpha':
			min_lam0, max_lam0 = 6400., 6839.
		else:
			continue
			
		igood = np.where((output_spec['ilam0'] >= min_lam0) & (output_spec['ilam0'] <= max_lam0))[0]
		if (len(igood)==0):
			continue

		## Save to file for GSF fitting ##
		f = open(f"/Users/guidorb/Dropbox/papers/z5_Templates/gsf/input_stacks/gsf_{filename}_{ext}.txt", "w")
		f.write('# lambda(AA) flux_nu eflux_nu_lines\n')
		f.write('# magzp = 31.4\n')
		for i,j,k in zip(output_spec['ilam'][igood],output_spec['iflux_corr'][igood],output_spec['istd_corr_maskedlines'][igood]):
			if i==0.:
				continue
			f.write("%0.9f %0.9f %0.9f\n"%(i,j,k))
		f.close()
	
	output_spec['ilam'] = output_spec['ilam'] / 10000.
	
	if plot==True:
		fig = plt.figure(figsize=(8.,4.5))
		ax = fig.add_subplot(111)
		ax.set_xlim(0.6,5.3)
		if ylims!=None:
			ax.set_ylim(ylims[0],ylims[1])
		ax.step(output_spec['ilam'], output_spec['iflux_scaled'], color='maroon', alpha=0.4, linewidth=1., label='NIRSpec spectrum (scaled only)')
		ax.step(output_spec['ilam'], output_spec['iflux_corr'], color='C0', alpha=0.8, linewidth=2., label='Final spectrum (scaled+slitloss corrected)')
		ax.step(output_spec['ilam'], output_spec['istd_corr'], color='black', linestyle='-', linewidth=1., alpha=0.2, label='16-84% stack error')
		ax.step(output_spec['ilam'], output_spec['iscatter_corr'], color='black', linestyle='--', linewidth=1., alpha=0.2, label='16-84% stack scatter')

		for filt in list(output_photo.keys()):
			ax.errorbar(output_photo[filt]['lam_cent'], output_photo[filt]['iflux_nircam'],
						xerr=output_photo[filt]['lam_err'], yerr=output_photo[filt]['istd_nircam'],
						marker='o', color='grey', ecolor='grey', markeredgecolor='white', markersize=12.5)
			ax.errorbar(output_photo[filt]['lam_cent'], output_photo[filt]['iflux_scaled'],
						marker='s', color='maroon', markeredgecolor='black', markersize=7.5, alpha=0.5)
			ax.errorbar(output_photo[filt]['lam_cent'], output_photo[filt]['iflux_corr'],
						marker='s', color='C0', markeredgecolor='black', markersize=7.5)
		ax.errorbar(1.e9, output_photo[filt]['iflux_nircam'],
					xerr=output_photo[filt]['lam_err'], yerr=output_photo[filt]['istd_nircam'],
					marker='o', color='grey', ecolor='grey', markeredgecolor='white', markersize=12.5, label='NIRCam photometry')
		ax.errorbar(1.e9, output_photo[filt]['iflux_scaled'],
					marker='s', color='maroon', markersize=7.5, alpha=0.5, label='NIRSpec photometry (scaled only)')
		ax.errorbar(1.e9, output_photo[filt]['iflux_corr'],
					marker='s', color='C0', markersize=7.5, label='NIRSpec photometry (scaled+slitloss corrected)')

		ax.plot(ax.set_xlim(),[0.,0.], linestyle='--', color='black', linewidth=1.)

		yval = ax.set_ylim()[1] - ((ax.set_ylim()[1]-ax.set_ylim()[0])/2.)
		iNgals_plot = iNgals / max(iNgals)
		iNgals_plot = iNgals_plot * yval
		ax.step(output_spec['ilam'], iNgals_plot, linewidth=1., color='green', label=r'Relative N$_{\rm gals}$ per $\lambda$ bin')
		ax.legend(loc='upper left', fontsize=8.5)
		gf.style_axes(ax, r'Observed Wavelength [$\mu$m]', 'Flux Density [nJy]')
		plt.tight_layout()
		plt.show()
	
	return output_spec, output_photo


def load_composite(filename, full_path=False):
	if full_path==True:
		output = pickle.load(open(filename, "rb"), encoding='latin1')
	else:
		output = pickle.load(open(f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/input_stacks/{filename}.p', "rb"), encoding='latin1')
	output_spec = output['spec']
	output_photo = output['phot']
	return output_spec, output_photo


def beta_slope(ilam, flux, err, z=None, mask_ciii=False, plot=False, lam_range=[1600.,3500.]):
	# Assume wavelength is in microns and fluxes are in microJy
	
	assert z!=None, 'Need a redshift to compute beta.'

	def fit_func(x, A, B): # this is your 'straight line' y=f(x)
		return A*x**B
	
	lam_A = ilam * 10000.    
	flux_flam = gf.fnu_to_flam(lam_A, flux*(1.e-23)*(1.e-6))
	err_flam = gf.fnu_to_flam(lam_A, err*(1.e-23)*(1.e-6))
	
	lo_lim, up_lim = lam_range[0]*(1.+z), lam_range[1]*(1.+z)
	idx_sel = np.where((lam_A > lo_lim) & (lam_A < up_lim))[0]
	
	ilo = gf.find_nearest(lam_A, lo_lim)
	iup = gf.find_nearest(lam_A, up_lim)

	bounds = ((-np.inf,-5.),(np.inf,0.))
	if np.unique(err_flam[idx_sel] == 0.):
		err_flam[idx_sel] = np.std(flux_flam[idx_sel])
		
	if mask_ciii==True:
		lo_lim, up_lim = 1850.*(1.+z), 1950.*(1.+z)
		idx_ciii = np.where((lam_A > lo_lim) & (lam_A < up_lim))[0]
		err_flam[idx_ciii] = 1.e10
		err[idx_ciii] = 1.e10
	
	popt, pcov = curve_fit(fit_func, lam_A[idx_sel], flux_flam[idx_sel], sigma=err_flam[idx_sel], absolute_sigma=True,
							p0=[1.e-8, -2.25], maxfev=10000000,
						   bounds=bounds)
	perr = np.sqrt(np.diag(pcov))
	yfit1 = fit_func(lam_A[idx_sel], *popt)
	
	# print('z=%0.3f beta=%0.3f+/-%0.3f' % (z, popt[-1], perr[-1]))

	if plot==True:
		fig = plt.figure(figsize=(10.,5.))
		ax = fig.add_subplot(111)
		ax.step(lam_A[idx_sel]/(1.+z), flux_flam[idx_sel], color='dodgerblue', linestyle='--', linewidth=2., zorder=2, label='NIRSpec spectrum')
		ax.plot(lam_A[idx_sel]/(1.+z), yfit1, color='darkorange', linestyle='--', linewidth=2., zorder=3, label='Best fit slope')
		ax.set_ylim(ax.set_ylim())
		ax.fill_between(lam_A[idx_sel]/(1.+z), flux_flam[idx_sel]-err_flam[idx_sel], flux_flam[idx_sel]+err_flam[idx_sel], color='dodgerblue', alpha=0.2, zorder=1, step='pre', label=r'1$\sigma$ error array')
		ax.legend(loc='upper right')
		gf.style_axes(ax, r'$\lambda_{\rm rest.}$ [$\AA$]', r'$F_{\lambda}$ [erg/s/cm$^{2}$/$\AA$]')
		plt.tight_layout()
		plt.savefig('/Users/guidorb/1181_38684_beta.pdf')
		plt.show()
	
	return popt[-1], perr[-1]


def fit_gsf_stack(filename, z_stack=None, fit_nebular=False):
	inputs = read_input('/Users/guidorb/AstroSoftware/gsf/param_files/gsf_highz_young.input')
	inputs['SPEC_FILE'] = f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/input_stacks/{filename}.txt'
	inputs['CAT_BB'] = '/Users/guidorb/Dropbox/papers/z5_Templates/gsf/input_stacks/nircam_gsf_1.txt'
	inputs['DIR_TEMP'] = '/Users/guidorb/AstroSoftware/gsf/templates/'
	inputs['DIR_OUT'] = '/Users/guidorb/Dropbox/papers/z5_Templates/gsf/output_stacks/'
	inputs['ID'] = '1'
	if fit_nebular==True:
		inputs['ADD_NEBULAE'] = 'y'
	elif fit_nebular==False:
		inputs['ADD_NEBULAE'] = 'n'
	inputs['DUST_MODEL'] = '3'
	inputs['NEBULAE_PRIOR'] = '0'
	
	inputs['ZMC'] = z_stack
	inputs['ZMCMIN'] = inputs['ZMC']-0.1
	inputs['ZMCMAX'] = inputs['ZMC']+0.1
	
	# Then, run template generate function;
	mb = run_gsf_template(inputs, fplt=1)
	
	mb.z_prior = np.arange(inputs['ZMCMIN'], inputs['ZMCMAX']+0.1, 0.1)
	mb.p_prior = np.ones_like(mb.z_prior)
	mb.neb_correlate = False
	
	# Since already z-fit done, we can skip z-fit;
	skip_fitz = True
	
	# Main;
	flag_suc = mb.main(cornerplot=True, specplot=0, sigz=1.0, ezmin=0.01, ferr=0, f_move=False, skip_fitz=skip_fitz)
	
	# Get SED and SFH output, averaging SFR over previous 10 Myrs;
	tset_SFR_SED = 0.01
	mmax = 1000
	
	dict_sfh = plot_sfh(mb, fil_path=mb.DIR_FILT, mmax=mmax,
						dust_model=mb.dust_model, DIR_TMP=mb.DIR_TMP,
						tset_SFR_SED=tset_SFR_SED, return_figure=False)
	
	dict_sed = plot_sed(mb, fil_path=mb.DIR_FILT, mmax=mmax,
						dust_model=mb.dust_model, DIR_TMP=mb.DIR_TMP,
						return_figure=False)
	
	output_files = glob.glob(inputs['DIR_OUT'] + '*1.*')
	for ofile in output_files:
		newfile = ofile.replace('1','z'+inputs['SPEC_FILE'].split('z')[-1].split('.')[0])
		os.system(f'mv {ofile} {newfile}')
		
	f = ascii.read(inputs['SPEC_FILE'])
	ilam, iflux, istd = f['lambda(AA)'], f['flux_nu'], f['eflux_nu_lines']
	ilam = ilam / 10000.
	if '/' in filename:
		filename = filename.split('/')[-1]
	
	fmodel = pyfits.open(inputs['DIR_OUT'] + f'gsf_spec_{filename}.fits')
	lam, model = fmodel[1].data['wave_model'], fmodel[1].data['f_model_noline_50'] * (1.e-19)
	model = gf.flam_to_fnu(lam, model) * (1.e23) * (1.e9)
	lam = lam / 10000.
	imodel = np.interp(ilam, lam, model)
	
	fig = plt.figure(figsize=(8.,4.))
	ax = fig.add_subplot(111)
	ax.step(ilam, iflux, zorder=2)
	ax.set_ylim()
	ax.fill_between(ilam, -istd, istd, alpha=0.2, zorder=1)
	ax.plot(ilam, imodel, color='darkred', zorder=3)
	ax.plot(ax.set_xlim(),[0.,0.], linestyle='--', color='black', linewidth=1.)
	gf.style_axes(ax, r'Observed Wavelength [$\mu$m]', 'Flux Density [nJy]')
	plt.tight_layout()
	plt.show()


def fit_gsf_single(name, add_nebulae='n', dust_model='3', input_dir='input', output_dir='output'):

	tab = ascii.read('/Users/guidorb/Dropbox/papers/z5_Templates/highz_galaxies_combined.dat')
	spectra = pickle.load(open('/Users/guidorb/Dropbox/papers/z5_Templates/spectra_final_03Nov2023.p', "rb"), encoding='latin1')

	if stack==False:
		i = np.where(tab['name'] == name)[0]
		z_spec = tab['z_spec'][i]
		ID = name.copy()
	elif stack==True:
		z_spec = float(name.split('_z')[-1].split('_')[0].replace('p','.'))
		ID = '1'

	## Fit a continuum model with GSF
	inputs = read_input('/Users/guidorb/AstroSoftware/gsf/param_files/gsf_highz.input')
	inputs['SPEC_FILE'] = f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/{input_dir}/nirspec_gsf_{name}.txt'
	inputs['CAT_BB'] = f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/{input_dir}/nircam_gsf_{name}.txt'
	inputs['DIR_TEMP'] = '/Users/guidorb/AstroSoftware/gsf/templates/'
	inputs['DIR_OUT'] = f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/{output_dir}/'
	inputs['ADD_NEBULAE'] = add_nebulae
	inputs['ID'] = ID
	# inputs['DUST_MODEL'] = dust_model

	inputs['ZMC'] = z_spec
	inputs['ZMCMIN'] = inputs['ZMC']-0.1
	inputs['ZMCMAX'] = inputs['ZMC']+0.1

	# Then, run template generate function;
	mb = run_gsf_template(inputs, fplt=1)
	
	mb.z_prior = np.arange(inputs['ZMCMIN'], inputs['ZMCMAX']+0.1, 0.1)
	mb.p_prior = np.ones_like(mb.z_prior)
	
	# Since already z-fit done, we can skip z-fit;
	skip_fitz = True
	
	# Main;
	flag_suc = mb.main(cornerplot=True, specplot=0, sigz=1.0, ezmin=0.01, ferr=0, f_move=False, skip_fitz=skip_fitz)
	
	# Get SED and SFH output, averaging SFR over previous 10 Myrs;
	tset_SFR_SED = 0.01
	mmax = 1000
	
	dict_sfh = plot_sfh(mb, fil_path=mb.DIR_FILT, mmax=mmax,
						dust_model=mb.dust_model, DIR_TMP=mb.DIR_TMP,
						tset_SFR_SED=tset_SFR_SED, return_figure=False)
	
	dict_sed = plot_sed(mb, fil_path=mb.DIR_FILT, mmax=mmax,
						dust_model=mb.dust_model, DIR_TMP=mb.DIR_TMP,
						return_figure=False)




def stack_1d(spectra, names, z_stack=None, AB_stack=None, method='median', apply_slitloss=True, clip=False, sigma_sig=3., bootstrap=False, flam=False, filename=None, save=['full','CIII','Balmer','OIIIHbeta','HeI','Halpha'], plot=False, ylims=None):

	tab = ascii.read('/Users/guidorb/Dropbox/Postdoc/highz_galaxies_combined.dat')

	assert filename is not None, "Need to specify file name including 'z[]p[]' keyword!"
	lam0_min, lam0_max = 0., 5.3
	ilam = np.arange(lam0_min*10000., lam0_max*10000.,10.)
	
	stack_flux = []
	redshifts = []
	stellar_masses = []
	for name in names:
		i = np.where(tab['name'] == name)[0]
		z_spec = tab['z_spec'][i]

		lam, spec, _, mask = af.get_prism_spectrum(spectra, name, clip=False)
		lam0 = lam / (1.+z_spec)
		fl = np.nanmean(spec[(lam0 > 0.15) & (lam0 < 0.2)])
		spec = spec / fl # normalize to unity

		spec[(mask == 1)] = 0.
		
		lam0 = lam0 * 10000.
		if z_stack > 0.:
			lam0 = lam0 * (1.+z_stack)
		
		ispec = np.interp(ilam, lam0, spec)
		stack_flux.append(ispec)
		redshifts.append(z_spec)

		try:
			f = pyfits.open(f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/output/SFH_{name}.fits')
			stellar_masses.append(float(f[0].header['Mstel_50']))
		except:
			pass

	stack_flux = np.array(stack_flux)
	redshifts = np.array(redshifts)
	stellar_masses = np.array(stellar_masses)
	
	ilam0 = ilam / (1.+z_stack)
	
	iflux = []
	istd = []
	ispread = []
	iNgals = []
	for i in range(len(stack_flux.T)):
		if clip==False:
			fluxes = stack_flux.T[i][(stack_flux.T[i] != 0.) & (np.isfinite(stack_flux.T[i])==True)]
			n = len(fluxes)
			iNgals.append(n)
			if n==0:
				iflux.append(0.)
				istd.append(0.)
				ispread.append(0.)
				continue
			
			if method=='median':
				iflux.append(np.median(fluxes))
			elif method=='mean':
				iflux.append(np.mean(fluxes))
				
			istd.append(np.mean([iflux[-1]-np.percentile(fluxes,16),np.percentile(fluxes,84)-iflux[-1]])/np.sqrt(n))
			ispread.append(np.mean([iflux[-1]-np.percentile(fluxes,16),np.percentile(fluxes,84)-iflux[-1]]))
				
			# if bootstrap==False:
			# 	istd.append(np.std(fluxes)/np.sqrt(n))
			# elif bootstrap==True:
			# 	stds = []
			# 	for N in range(100):
			# 		idx = []
			# 		while len(idx) < int(n*0.75):
			# 			r = random.randint(0,n-1)
			# 			if r not in idx:
			# 				idx.append(r)
			# 		idx = np.array(idx).astype(int)
			# 		stds.append(np.std(fluxes[idx]))
			# 	stds = np.array(stds)
			# 	stds = np.sqrt(np.median(stds**2.))
			# 	istd.append(stds)
				
				
		elif clip==True:
			flux_arr, minval, maxval, ival = custom_sigma_clip(stack_flux.T[i][(stack_flux.T[i] != 0.) & (np.isfinite(stack_flux.T[i])==True)], low=sigma_sig, high=sigma_sig, op=method)
			n = len(flux_arr)
			iNgals.append(n)
			if n==0:
				iflux.append(0.)
				istd.append(0.)
				ispread.append([0.,0.])
				continue
			
			if method=='median':
				iflux.append(np.median(flux_arr))
			elif method=='mean':
				iflux.append(np.mean(flux_arr))
				
			istd.append(np.mean([iflux[-1]-np.percentile(fluxes,16),np.percentile(fluxes,84)-iflux[-1]])/np.sqrt(n))
			ispread.append(np.mean([iflux[-1]-np.percentile(fluxes,16),np.percentile(fluxes,84)-iflux[-1]]))
				
			# if bootstrap==False:
			# 	istd.append(np.std(flux_arr)/np.sqrt(n))
			# elif bootstrap==True:
			# 	stds = []
			# 	for N in range(100):
			# 		idx = []
			# 		while len(idx) < int(n*0.75):
			# 			r = random.randint(0,n-1)
			# 			if r not in idx:
			# 				idx.append(r)
			# 		idx = np.array(idx).astype(int)
			# 		stds.append(np.std(flux_arr[idx]))
			# 	stds = np.array(stds)
			# 	stds = np.sqrt(np.median(stds**2.))
			# 	istd.append(stds)
				
	iflux = np.array(iflux)
	istd = np.array(istd)
	ispread = np.array(ispread)
	iNgals = np.array(iNgals)

	# Correct for wavelength-dependent slit loss
	if apply_slitloss==True:
		iflux = af.correct_slitloss(ilam/10000., iflux)
		istd = af.correct_slitloss(ilam/10000., istd)
		ispread = af.correct_slitloss(ilam/10000., ispread)
	
	# Scale for overall slit loss
	normflux = gf.AB_to_flux(AB_stack, output_unit='jy') * (1.e9)
	specflux = np.median(iflux[(ilam0 > 1500.) & (ilam0 < 2000.)])
	factor = normflux/specflux
	
	if AB_stack > 0.:
		iflux *= factor
		istd *= factor
		ispread *= factor
	
	specflux = ufloat(np.nanmedian(iflux[(ilam0 > 1500.) & (ilam0 < 2000.)]), np.nanstd(iflux[(ilam0 > 1500.) & (ilam0 < 2000.)]))
	MUV = gf.M_from_m(gf.flux_to_AB(unp.nominal_values(specflux) * (1.e-9), flux_err=unp.std_devs(specflux)*(1.e-9)), z=z_stack)
	logMstar = unp.uarray(np.nanmedian(stellar_masses),np.nanstd(stellar_masses))
	# if filename!=None:
	# 	print(filename, len(names), '%0.2f$\pm$%0.2f'%(z_stack,np.std(redshifts)), '%0.2f$\pm$%0.2f'%(unp.nominal_values(MUV),unp.std_devs(MUV)), '%0.2f$\pm$%0.2f'%(unp.nominal_values(logMstar),unp.std_devs(logMstar)))
	# else:
	# 	print(z_stack, MUV)
	
	imask = af.generate_line_mask(ilam, z_spec=z_stack, dv=4000)
	istd_lines = istd.copy()
	istd_lines[(imask == 1)] = 1.e20
	
	for ext in save:
# 		if ext == 'full':
# 			min_lam0, max_lam0 = 0., 20000.
# 		elif ext == 'CIII':
# #             min_lam0, max_lam0 = 1750., 2125.
# 			# min_lam0, max_lam0 = 1400., 2125.
# 			min_lam0, max_lam0 = 1400., 2125.
# 		elif ext == 'Balmer':
# 			min_lam0, max_lam0 = 3375., 4605.
# 		elif ext == 'OIIIHbeta':
# #             min_lam0, max_lam0 = 4605., 5325.
# 			min_lam0, max_lam0 = 4550., 5325.
# 		elif ext == 'HeI':
# 			min_lam0, max_lam0 = 5750., 6000.
# 		elif ext == 'Halpha':
# 			min_lam0, max_lam0 = 6400., 6839.
# 		else:
# 			continue
			
		if ext == 'full':
			min_lam0, max_lam0 = 0., 20000.
		elif ext == 'CIII':
			min_lam0, max_lam0 = 1550., 2300.
		elif ext == 'Balmer':
			min_lam0, max_lam0 = 3200., 4800.
		elif ext == 'OIIIHbeta':
			min_lam0, max_lam0 = 4550., 5325.
		elif ext == 'HeI':
			min_lam0, max_lam0 = 5750., 6000.
		elif ext == 'Halpha':
			min_lam0, max_lam0 = 6400., 6839.
		else:
			continue


		igood = np.where((ilam0 >= min_lam0) & (ilam0 <= max_lam0))[0]
		# if ((len(igood)==0) | (min(ilam0) > min_lam0) | (max(ilam0) < max_lam0)) & (ext!='full'):
		if (len(igood)==0) & (ext!='full'):
			continue

		## Save to file for GSF fitting ##
		f = open(f"/Users/guidorb/Dropbox/papers/z5_Templates/gsf/input_stacks/{filename}_{ext}.txt", "w")
		f.write('# lambda(AA) flux_nu eflux_nu eflux_nu_lines eflux_nu_spread\n')
		f.write('# magzp = 31.4\n')
		for i,j,k,l,m in zip(ilam[igood],iflux[igood],istd[igood],istd_lines[igood],ispread[igood]):
			if i==0.:
				continue
			f.write("%0.9f %0.9f %0.9f %0.9f %0.9f\n"%(i,j,l,k,m))
		f.close()
	
	if plot==True:

		fig = plt.figure(figsize=(8.,4.))
		ax = fig.add_subplot(111)
		ax.step(ilam/10000., iflux)
		ax.fill_between(ilam/10000., -istd, istd, alpha=0.2)
		ax.step(ilam/10000., ispread, color='teal', alpha=0.4)
		ax.plot(ax.set_xlim(),[0.,0.], linestyle='--', color='black', linewidth=1.)

		yval = ax.set_ylim()[1] - ((ax.set_ylim()[1]-ax.set_ylim()[0])/4.)
		iNgals_plot = iNgals / max(iNgals)
		iNgals_plot = iNgals_plot * yval
		ax.step(ilam/10000., iNgals_plot, linewidth=2., color='darkred')
		if ylims!=None:
			ax.set_ylim(ylims[0],ylims[1])
		gf.style_axes(ax, r'Observed Wavelength [$\mu$m]', 'Flux Density [nJy]')
		plt.tight_layout()
		plt.show()
	
	return ilam, iflux, istd, iNgals





def fit_emission_lines(filename, z_stack=None, Av=0.):
	
	lines = {'CIV':1549.4795,
			 'HEII_1':1640.420,
			 'OIII_05':1663.4795,
			 'CIII':1908.734,
			 
			 # 'OII_UV_1':3727.092,
			 # 'OII_UV_2':3729.875,
			 'OII_UV_1':np.mean([3727.092,3729.875]),
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
	

	inputs = {
			  'SPEC_FILE' : f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/input_stacks/{filename}.txt',
			 }
	if '/' in filename:
		filename = filename.split('/')[-1]
	ext = filename.split('_')[-1]
	root = filename.replace('_'+ext,'')
		
	inputs['MODEL_FILE'] = f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/output_stacks/gsf_spec_{filename}.fits'
	inputs['SFH_FILE'] = f'/Users/guidorb/Dropbox/papers/z5_Templates/gsf/output_stacks/SFH_{filename}.fits'
	
	input_spec = ascii.read(inputs['SPEC_FILE'])
	ilam, iflux, istd = input_spec['lambda(AA)'], input_spec['flux_nu'], input_spec['eflux_nu_lines']
	spec = gf.fnu_to_flam(ilam, iflux*(1.e-9)*(1.e-23), fnu_err=istd*(1.e-9)*(1.e-23))
	
	SCALE_FACTOR = 1.e-21
	
	iflux = unp.nominal_values(spec) / SCALE_FACTOR
	istd = unp.std_devs(spec) / SCALE_FACTOR
	
	output_spec = pyfits.open(inputs['MODEL_FILE'])
	lam, model, model_low, model_high = output_spec[1].data['wave_model'], output_spec[1].data['f_model_noline_50']*float(output_spec[1].header['SCALE']), output_spec[1].data['f_model_noline_16']*float(output_spec[1].header['SCALE']), output_spec[1].data['f_model_noline_84']*float(output_spec[1].header['SCALE'])
	imodel = np.interp(ilam, lam, model) / SCALE_FACTOR
	imodel_low = np.interp(ilam, lam, model_low) / SCALE_FACTOR
	imodel_high = np.interp(ilam, lam, model_high) / SCALE_FACTOR
	
	ilam0 = ilam / (1.+z_stack)
	idx = np.where(
					(istd != 0.) & 
					(imodel != 0.) & 
					(iflux != 0.) & 
					(np.isfinite(istd)==True) & 
					(np.isfinite(imodel)==True) & 
					(np.isfinite(iflux)==True)
					)[0]
	
	iflux = iflux[idx]
	istd = istd[idx]
	imodel = imodel[idx]
	
	imodel_low = imodel_low[idx]
	imodel_high = imodel_high[idx]
	imodel_err = np.mean([imodel_high-imodel,imodel-imodel_low], axis=0)
	ilam0 = ilam0[idx]
	ilam = ilam[idx]
	ires = iflux-imodel
	
	# if 'Balmer' in filename:
	# 	ires = ires - np.nanmean(ires[(ilam0 < 3600.) | (ilam0 > 4450.)])
	# elif 'OIIIHbeta' in filename:
	# 	ires = ires - np.nanmean(ires[(ilam0 < 4650.) | (ilam0 > 5150.)])

	# if 'OIIIHbeta' in filename:
	# 	print(np.nanmean(ires[(ilam0 > 4600.) & (ilam0 < 4650.)]))
	# 	ires = ires + np.nanmean(ires[(ilam0 > 4600.) & (ilam0 < 4650.)])
	# 	ires = ires + 0.05
	# 	fig = plt.figure(figsize=(8.,5.))
	# 	ax = fig.add_subplot(111)
	# 	ax.step(ilam, ires)
	# 	ax.plot(ax.set_xlim(), [0.,0.])
	# 	# ax.set_xlim(30000.,34000.)
	# 	# ax.set_ylim(-0.5,0.5)
	# 	plt.tight_layout()
	# 	plt.show()

	if len(np.unique(istd))==1:
		istd = np.ones_like(iflux) * np.nanstd(iflux)
	
	# De-redden spectra and models
	if Av>0.:
		Alam = extinction.ccm89(ilam0, a_v=Av, r_v=3.1)
		iflux = iflux / (10**(-0.4*Alam))
		imodel = imodel / (10**(-0.4*Alam))
		ires = ires / (10**(-0.4*Alam))
	 
	xfit, yfit, efit, mfit, mfit_err = ilam.copy(), ires.copy(), istd.copy(), imodel.copy(), imodel_err.copy()

	best_params_all, best_eparams_all = [], []
	if 'CIII' in filename:
		lines_to_evaluate = ['CIV','HEII_1','CIII']
		Nlines = 3
		def log_likelihood(theta, x, y, yerr):
			x1, x2, x3, A1, A2, A3, sig1, sig2, sig3 = theta
			model1 = (10**A1) * np.exp(-0.5*(((x-x1)**2.)/(2*(10**sig1)**2.)))
			model2 = (10**A2) * np.exp(-0.5*(((x-x2)**2.)/(2*(10**sig2)**2.)))
			model3 = (10**A3) * np.exp(-0.5*(((x-x3)**2.)/(2*(10**sig3)**2.)))
			model = np.sum((model1,
							model2,
							model3), axis=0)
			loglike = -0.5*np.sum(((y - model)/yerr)**2)
			return loglike
		def log_prior(theta):
			x1, x2, x3, A1, A2, A3, sig1, sig2, sig3 = theta
			if (((lines['CIV']*(1.+z_stack))-100.) < x1 < ((lines['CIV']*(1.+z_stack))+100.)) & \
				(((lines['HEII_1']*(1.+z_stack))-100.) < x2 < ((lines['HEII_1']*(1.+z_stack))+100.)) & \
				(((lines['CIII']*(1.+z_stack))-100.) < x3 < ((lines['CIII']*(1.+z_stack))+100.)) & \
				(-5. < A1 < np.log10(np.nanmax(yfit[(xfit>((lines['CIV']-25.)*(1.+z_stack))) & (xfit<((lines['CIV']+25.)*(1.+z_stack)))]))) & \
				(-5. < A2 < np.log10(np.nanmax(yfit[(xfit>((lines['HEII_1']-25.)*(1.+z_stack))) & (xfit<((lines['HEII_1']+25.)*(1.+z_stack)))]))) & \
				(-5. < A3 < np.log10(np.nanmax(yfit[(xfit>((lines['CIII']-25.)*(1.+z_stack))) & (xfit<((lines['CIII']+25.)*(1.+z_stack)))]))) & \
				(-3. < sig1 < 0.) & \
				(-3. < sig2 < 0.) & \
				(-3. < sig3 < 0.):
				return 0.0
			return -1000.
		initial = [lines['CIV']*(1.+z_stack),
				   lines['HEII_1']*(1.+z_stack),
				   lines['CIII']*(1.+z_stack),
				   np.log10(np.nanmax(yfit[(xfit>((lines['CIV']-25.)*(1.+z_stack))) & (xfit<((lines['CIV']+25.)*(1.+z_stack)))])),
				   np.log10(np.nanmax(yfit[(xfit>((lines['HEII_1']-25.)*(1.+z_stack))) & (xfit<((lines['HEII_1']+25.)*(1.+z_stack)))])),
				   np.log10(np.nanmax(yfit[(xfit>((lines['CIII']-25.)*(1.+z_stack))) & (xfit<((lines['CIII']+25.)*(1.+z_stack)))])),
				   np.log10(25.),
				   np.log10(25.),
				   np.log10(25.)]
			
	elif 'Balmer' in filename:
		lines_to_evaluate = ['OII_UV_1','NEIII_UV_1','NEIII_UV_2','HDELTA','HGAMMA','HEI_1']
		Nlines = 6
		def log_likelihood(theta, x, y, yerr):
			x1, x2, x3, x4, x5, x6, A1, A2, A3, A4, A5, A6, sig1, sig2, sig3, sig4, sig5, sig6 = theta
			model1 = (10**A1) * np.exp(-0.5*(((x-x1)**2.)/(2*(10**sig1)**2.)))
			model2 = (10**A2) * np.exp(-0.5*(((x-x2)**2.)/(2*(10**sig2)**2.)))
			model3 = (10**A3) * np.exp(-0.5*(((x-x3)**2.)/(2*(10**sig3)**2.)))
			model4 = (10**A4) * np.exp(-0.5*(((x-x4)**2.)/(2*(10**sig4)**2.)))
			model5 = (10**A5) * np.exp(-0.5*(((x-x5)**2.)/(2*(10**sig5)**2.)))
			model6 = (10**A6) * np.exp(-0.5*(((x-x6)**2.)/(2*(10**sig6)**2.)))
			model = np.sum((model1,
							model2,
							model3,
							model4,
							model5,
							model6), axis=0)
			loglike = -0.5*np.sum(((y - model)/yerr)**2)
			return loglike
		def log_prior(theta):
			x1, x2, x3, x4, x5, x6, A1, A2, A3, A4, A5, A6, sig1, sig2, sig3, sig4, sig5, sig6 = theta

			if (((lines['OII_UV_1']*(1.+z_stack))-100.) < x1 < ((lines['OII_UV_1']*(1.+z_stack))+100.)) & \
				(((lines['NEIII_UV_1']*(1.+z_stack))-100.) < x2 < ((lines['NEIII_UV_1']*(1.+z_stack))+100.)) & \
				(((lines['NEIII_UV_2']*(1.+z_stack))-100.) < x3 < ((lines['NEIII_UV_2']*(1.+z_stack))+100.)) & \
				(((lines['HDELTA']*(1.+z_stack))-100.) < x4 < ((lines['HDELTA']*(1.+z_stack))+100.)) & \
				(((lines['HGAMMA']*(1.+z_stack))-100.) < x5 < ((lines['HGAMMA']*(1.+z_stack))+100.)) & \
				(((lines['HEI_1']*(1.+z_stack))-100.) < x6 < ((lines['HEI_1']*(1.+z_stack))+100.)) & \
				(-5. < A1 < np.log10(np.nanmax(yfit[(xfit>((lines['OII_UV_1']-25.)*(1.+z_stack))) & (xfit<((lines['OII_UV_1']+25.)*(1.+z_stack)))]))) & \
				(-5. < A2 < np.log10(np.nanmax(yfit[(xfit>((lines['NEIII_UV_1']-25.)*(1.+z_stack))) & (xfit<((lines['NEIII_UV_1']+25.)*(1.+z_stack)))]))) & \
				(-5. < A3 < np.log10(np.nanmax(yfit[(xfit>((lines['NEIII_UV_2']-25.)*(1.+z_stack))) & (xfit<((lines['NEIII_UV_2']+25.)*(1.+z_stack)))]))) & \
				(-5. < A4 < np.log10(np.nanmax(yfit[(xfit>((lines['HDELTA']-25.)*(1.+z_stack))) & (xfit<((lines['HDELTA']+25.)*(1.+z_stack)))]))) & \
				(-5. < A5 < np.log10(np.nanmax(yfit[(xfit>((lines['HGAMMA']-25.)*(1.+z_stack))) & (xfit<((lines['HGAMMA']+25.)*(1.+z_stack)))]))) & \
				(-5. < A6 < np.log10(np.nanmax(yfit[(xfit>((lines['HEI_1']-25.)*(1.+z_stack))) & (xfit<((lines['HEI_1']+25.)*(1.+z_stack)))]))) & \
				(-3. < sig1 < 1.) & \
				(-3. < sig2 < 1.) & \
				(-3. < sig3 < 1.) & \
				(-3. < sig4 < 1.) & \
				(-3. < sig5 < 1.) & \
				(-3. < sig6 < 1.):
				return 0.0
			return -np.inf
		initial = [lines['OII_UV_1']*(1.+z_stack),
					lines['NEIII_UV_1']*(1.+z_stack),
					lines['NEIII_UV_2']*(1.+z_stack),
					lines['HDELTA']*(1.+z_stack),
					lines['HGAMMA']*(1.+z_stack),
					lines['HEI_1']*(1.+z_stack),
					np.log10(np.nanmax(yfit[(xfit>((lines['OII_UV_1']-25.)*(1.+z_stack))) & (xfit<((lines['OII_UV_1']+25.)*(1.+z_stack)))])),
					np.log10(np.nanmax(yfit[(xfit>((lines['NEIII_UV_1']-25.)*(1.+z_stack))) & (xfit<((lines['NEIII_UV_1']+25.)*(1.+z_stack)))])),
					np.log10(np.nanmax(yfit[(xfit>((lines['NEIII_UV_2']-25.)*(1.+z_stack))) & (xfit<((lines['NEIII_UV_2']+25.)*(1.+z_stack)))])),
					np.log10(np.nanmax(yfit[(xfit>((lines['HDELTA']-25.)*(1.+z_stack))) & (xfit<((lines['HDELTA']+25.)*(1.+z_stack)))])),
					np.log10(np.nanmax(yfit[(xfit>((lines['HGAMMA']-25.)*(1.+z_stack))) & (xfit<((lines['HGAMMA']+25.)*(1.+z_stack)))])),
					np.log10(np.nanmax(yfit[(xfit>((lines['HEI_1']-25.)*(1.+z_stack))) & (xfit<((lines['HEI_1']+25.)*(1.+z_stack)))])),
					np.log10(50.),
					np.log10(50.),
					np.log10(50.),
					np.log10(50.),
					np.log10(50.),
					np.log10(50.)]
			
	elif 'OIIIHbeta' in filename:
		lines_to_evaluate = ['HBETA','OIII_2','OIII_3','HEII_2']
		Nlines = 4
		def log_likelihood(theta, x, y, yerr):
			x1, x2, x3, x4, A1, A2, A3, A4, sig1, sig2, sig3, sig4 = theta
			model1 = (10**A1) * np.exp(-0.5*(((x-x1)**2.)/(2*(10**sig1)**2.)))
			model2 = (10**A2) * np.exp(-0.5*(((x-x2)**2.)/(2*(10**sig2)**2.)))
			model3 = (10**A3) * np.exp(-0.5*(((x-x3)**2.)/(2*(10**sig3)**2.)))
			model4 = (10**A4) * np.exp(-0.5*(((x-x4)**2.)/(2*(10**sig4)**2.)))
			model = np.sum((model1,
							model2,
							model3,
							model4), axis=0)
			loglike = -0.5*np.sum(((y - model)/yerr)**2)
			return loglike
		def log_prior(theta):
			x1, x2, x3, x4, A1, A2, A3, A4, sig1, sig2, sig3, sig4 = theta
			if (((lines['HBETA']*(1.+z_stack))-100.) < x1 < ((lines['HBETA']*(1.+z_stack))+100.)) & \
				(((lines['OIII_2']*(1.+z_stack))-100.) < x2 < ((lines['OIII_2']*(1.+z_stack))+100.)) & \
				(((lines['OIII_3']*(1.+z_stack))-100.) < x3 < ((lines['OIII_3']*(1.+z_stack))+100.)) & \
				(((lines['HEII_2']*(1.+z_stack))-100.) < x4 < ((lines['HEII_2']*(1.+z_stack))+100.)) & \
				(-5. < A1 < np.log10(np.nanmax(yfit[(xfit>((lines['HBETA']-25.)*(1.+z_stack))) & (xfit<((lines['HBETA']+25.)*(1.+z_stack)))]))) & \
				(-5. < A2 < np.log10(np.nanmax(yfit[(xfit>((lines['OIII_2']-25.)*(1.+z_stack))) & (xfit<((lines['OIII_2']+25.)*(1.+z_stack)))]))) & \
				(-5. < A3 < np.log10(np.nanmax(yfit[(xfit>((lines['OIII_3']-25.)*(1.+z_stack))) & (xfit<((lines['OIII_3']+25.)*(1.+z_stack)))]))) & \
				(-5. < A4 < np.log10(np.nanmax(yfit[(xfit>((lines['HEII_2']-25.)*(1.+z_stack))) & (xfit<((lines['HEII_2']+25.)*(1.+z_stack)))]))) & \
				(-3. < sig1 < 1.) & \
				(-3. < sig2 < 1.) & \
				(-3. < sig3 < 1.) & \
				(-3. < sig4 < 1.):
				return 0.0
			return -1000.
		initial = [lines['HBETA']*(1.+z_stack),
				   lines['OIII_2']*(1.+z_stack),
				   lines['OIII_3']*(1.+z_stack),
				   lines['HEII_2']*(1.+z_stack),
							 np.log10(np.nanmax(yfit[(xfit>((lines['HBETA']-25.)*(1.+z_stack))) & (xfit<((lines['HBETA']+25.)*(1.+z_stack)))])),
				   np.log10(np.nanmax(yfit[(xfit>((lines['OIII_2']-25.)*(1.+z_stack))) & (xfit<((lines['OIII_2']+25.)*(1.+z_stack)))])),
				   np.log10(np.nanmax(yfit[(xfit>((lines['OIII_3']-25.)*(1.+z_stack))) & (xfit<((lines['OIII_3']+25.)*(1.+z_stack)))])),
				   np.log10(np.nanmax(yfit[(xfit>((lines['HEII_2']-25.)*(1.+z_stack))) & (xfit<((lines['HEII_2']+25.)*(1.+z_stack)))])),
				   np.log10(50.),
				   np.log10(50.),
				   np.log10(50.),
				   np.log10(50.)]
			
	elif 'HeI' in filename:
		lines_to_evaluate = ['HEI']
		Nlines = 1
		def log_likelihood(theta, x, y, yerr):
			x1, A1, sig1 = theta
			model = (10**A1) * np.exp(-0.5*(((x-x1)**2.)/(2*(10**sig1)**2.)))
			loglike = -0.5*np.sum(((y - model)/yerr)**2)
			return loglike
		def log_prior(theta):
			x1, A1, sig1 = theta
			if (((lines['HEI']*(1.+z_stack))-100.) < x1 < ((lines['HEI']*(1.+z_stack))+100.)) & \
				  (-1. < A1 < np.log10(np.nanmax(yfit[(xfit>((lines['HEI']-25.)*(1.+z_stack))) & (xfit<((lines['HEI']+25.)*(1.+z_stack)))]))) & \
				(-3. < sig1 < 2.):
				return 0.0
			return -1000.
		initial = [lines['HEI']*(1.+z_stack),
							 np.log10(np.nanmax(yfit[(xfit>((lines['HEI']-25.)*(1.+z_stack))) & (xfit<((lines['HEI']+25.)*(1.+z_stack)))])),
				   np.log10(50.)]
	
	elif 'Halpha' in filename:
		lines_to_evaluate = ['HALPHA','SII_1']
		Nlines = 2
		def log_likelihood(theta, x, y, yerr):
			x1, x2, A1, A2, sig1, sig2 = theta
			model1 = (10**A1) * np.exp(-0.5*(((x-x1)**2.)/(2*(10**sig1)**2.)))
			model2 = (10**A2) * np.exp(-0.5*(((x-x2)**2.)/(2*(10**sig2)**2.)))
			model = np.sum((model1,
							model2), axis=0)
			loglike = -0.5*np.sum(((y - model)/yerr)**2)
			return loglike
		def log_prior(theta):
			x1, x2, A1, A2, sig1, sig2 = theta
			if (((lines['HALPHA']*(1.+z_stack))-100.) < x1 < ((lines['HALPHA']*(1.+z_stack))+100.)) & \
				 (((lines['SII_1']*(1.+z_stack))-100.) < x2 < ((lines['SII_1']*(1.+z_stack))+100.)) & \
				 (-5. < A1 < np.log10(np.nanmax(yfit[(xfit>((lines['HALPHA']-25.)*(1.+z_stack))) & (xfit<((lines['HALPHA']+25.)*(1.+z_stack)))]))) & \
				(-5. < A2 < np.log10(np.nanmax(yfit[(xfit>((lines['SII_1']-25.)*(1.+z_stack))) & (xfit<((lines['SII_1']+25.)*(1.+z_stack)))]))) & \
				(-3. < sig1 < 1.) & \
				(-3. < sig2 < 1.):
				return 0.0
			return -1000.
		initial = [lines['HALPHA']*(1.+z_stack),
							 lines['SII_1']*(1.+z_stack),
							 np.log10(np.nanmax(yfit[(xfit>((lines['HALPHA']-25.)*(1.+z_stack))) & (xfit<((lines['HALPHA']+25.)*(1.+z_stack)))])),
				   np.log10(np.nanmax(yfit[(xfit>((lines['SII_1']-25.)*(1.+z_stack))) & (xfit<((lines['SII_1']+25.)*(1.+z_stack)))])),
				   np.log10(50.),
				   np.log10(50.)]
	
	Nparams = Nlines * 3
	def log_probability(theta, x, y, yerr):
		lp = log_prior(theta)
		loglike = log_likelihood(theta, x, y, yerr)
		if (np.isfinite(lp)==False) | (np.isfinite(loglike)==False):
			return -np.inf
		return lp + loglike
	
	nll = lambda *args: -log_likelihood(*args)
	
	soln = minimize(nll, initial, args=(ilam, ires, istd))
	
	pos = soln.x + 1e-4 * np.random.randn(Nparams*25, Nparams)
	nwalkers, ndim = pos.shape
	
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(ilam, ires, istd))
	sampler.run_mcmc(pos, 500, progress=True)
	
	flat_samples = sampler.get_chain(discard=50, thin=15, flat=True)
	best_params = []
	best_params_err = []
	for i in range(ndim):
		mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
		q = np.diff(mcmc)
		best_params.append(mcmc[1])
		best_params_err.append(np.mean(q))
	
	def gauss(x, x0, A, sig):
		g = A * unp.exp(-0.5*(((x-x0)**2.)/(2*sig**2.)))
		return g
	

	fig = plt.figure(figsize=(8.,4.))
	ax = fig.add_subplot(111)
	ax.step(xfit, yfit, color='blue', linewidth=1.5, label='NIRSpec stack', zorder=1)
	ax.set_ylim()
	ax.fill_between(xfit, -efit, efit, color='blue', alpha=0.1, label=r'Stack $1\sigma$', zorder=1)
	ax.step(xfit, -efit, linewidth=0.5, linestyle='-', color='blue', alpha=0.2, zorder=1)
	ax.step(xfit, efit, linewidth=0.5, linestyle='-', color='blue', alpha=0.2, zorder=1)
	ax.plot(ax.set_xlim(), [0.,0.], linestyle='--', linewidth=1., color='black')
		
	tot_model = np.zeros_like(xfit)
	EWs = ufloat(0.,0.)
	cards = []
	print(filename)
	print('Line   x0   A   Sigma   EW0 [A]   Flux [1E-21]')
	for nline, line in enumerate(lines_to_evaluate):
		l = lines[line] * (1.+z_stack)
		
		# Construct profile from best-fit parameters (incl. uncertainties)
		x0 = ufloat(best_params[0+nline], best_params_err[0+nline])
		A = 10**ufloat(best_params[len(lines_to_evaluate)+nline], best_params_err[len(lines_to_evaluate)+nline])
		sig = 10**ufloat(best_params[len(lines_to_evaluate)*2 + nline], best_params_err[len(lines_to_evaluate)*2 + nline])
		y = gauss(xfit, x0, A, sig)
		
		# Plot best fit model
		ax.plot(xfit, unp.nominal_values(y), color='darkorange', linewidth=1.5, zorder=3)
		tot_model = tot_model + unp.nominal_values(y)
		
		# Continuum model
		ymod = unp.uarray(mfit,mfit_err)
		
		# Calculate EW
		dx = np.diff(xfit)
		dx = np.concatenate([dx,[dx[-1]]])
		line_flux = np.sum(y*dx)
		C = np.nanmedian(ymod[(xfit > (l-sig)) & (xfit < (l+sig))])
		
		EW = line_flux / C
		EW0 = EW / (1.+z_stack)
		if np.isfinite(EW.n)==False:
			continue
		print(line, '%0.2f+/-%0.2f %0.2f+/-%0.2f %0.2f+/-%0.2f %0.2f+/-%0.2f %0.2f+/-%0.2f'%(x0.n,x0.s,A.n,A.s,sig.n,sig.s,EW0.n,EW0.s,line_flux.n,line_flux.s))
		
		line = line.upper()
		if Av>0.:
			dust = '_DUSTCORR'
		else:
			dust = ''

		c1 = pyfits.Card(f'LAM_{line}'+dust, x0.n, f'Lam_obs for {line} [Ang]')
		c2 = pyfits.Card(f'LAM_ERR_{line}'+dust, x0.s, f'Lam_obs for {line} [Ang]')

		c3 = pyfits.Card(f'LAM0_{line}'+dust, x0.n, f'Lam_rest for {line} [Ang]')

		c4 = pyfits.Card(f'A_{line}'+dust, A.n, f'A for {line} [1E-21 cgs]')
		c5 = pyfits.Card(f'A_ERR_{line}'+dust, A.s, f'A err for {line} [1E-21 cgs]')
		
		c6 = pyfits.Card(f'SIGMA_{line}'+dust, sig.n, f'Sig for {line} [Ang]')
		c7 = pyfits.Card(f'SIGMA_ERR_{line}'+dust, sig.s, f'Sig err for {line} [Ang]')

		c8 = pyfits.Card(f'FLUX_{line}'+dust, line_flux.n, f'Flux for {line} [1E-21 cgs')
		c9 = pyfits.Card(f'FLUX_ERR_{line}'+dust, line_flux.s, f'Flux err for {line} [1E-21 cgs]')

		c10 = pyfits.Card(f'EW_{line}'+dust, EW.n, f'EW {line} [Ang]')
		c11 = pyfits.Card(f'EW_ERR_{line}'+dust, EW.s, f'EW err for {line} [Ang]')
		
		c12 = pyfits.Card(f'EW0_{line}'+dust, EW0.n, f'EW0 {line} [Ang]')
		c13 = pyfits.Card(f'EW0_ERR_{line}'+dust, EW0.s, f'EW0 err for {line} [Ang]')
		
		cards.append(c1)
		cards.append(c2)
		cards.append(c3)
		cards.append(c4)
		cards.append(c5)
		cards.append(c6)
		cards.append(c7)
		cards.append(c8)
		cards.append(c9)
		cards.append(c10)
		cards.append(c11)
		cards.append(c12)
		cards.append(c13)
		
	ax.plot(xfit, tot_model, color='darkred', linewidth=1., zorder=200000)
	gf.style_axes(ax, 'Observed Wavelength', 'Flux Density')
	plt.show()
		
	return best_params, best_params_err, cards