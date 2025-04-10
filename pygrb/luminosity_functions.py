from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy.integrate import trapezoid
import numpy as np
from . import general_functions as gf
import matplotlib.pyplot as plt

def MUV_to_LUV(MUV):
	LUV = (2.5e29) * (10**((MUV+21.91)/-2.5))
	return LUV

def LUV_to_MUV(LUV):
	MUV = -21.91 - 2.5*np.log10(LUV / (2.5e29))
	return MUV

class Schechter:
	def __init__(self, phi_star, Mstar, alpha):
		if phi_star < 0.:
			self.phi_star = 10**phi_star
		else:
			self.phi_star = phi_star
		self.Mstar = Mstar
		self.alpha = alpha

	def get_phi_mag(self, MUV_array=None):
		if MUV_array==None:
			MUV_array = np.arange(-25.,-14.99,0.01)
		phi_mag = (unp.log(10.)/2.5) * self.phi_star * ((10**(0.4*(self.Mstar-MUV_array)))**(self.alpha+1.)) * unp.exp(-10.**(0.4*(self.Mstar-MUV_array)))
		return MUV_array, unp.nominal_values(phi_mag), unp.std_devs(phi_mag)

	def get_number_counts(self, Mint=None):
		assert Mint!=None, 'Please provide a magnitude to integrate the UVLF down to.'
		x, y, yerr = self.get_phi_mag()
		
		pUV = trapezoid(x[(x <= Mint)], y[(x <= Mint)])
		return pUV

	def get_puv(self, Mint=None, log=False):
		if Mint==None:
			Mint = -18.
		
		x, y, yerr = self.get_phi_mag()
		xlum = MUV_to_LUV(x)
		
		pUV_upp = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]+yerr[(x <= Mint)])
		pUV = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)])
		pUV_low = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]-yerr[(x <= Mint)])
		pUV_err = np.mean(abs(np.diff([pUV_upp, pUV, pUV_low])))

		if log==True:
			pUV = unp.log10(ufloat(pUV, pUV_err))
			pUV_err = unp.std_devs(pUV)
			pUV = unp.nominal_values(pUV)

		pUV = ufloat(pUV, pUV_err)
		return pUV

	def get_psfr(self, Mint=None, log=False):
		if Mint==None:
			Mint = -18.

		pUV = self.get_puv(Mint=Mint, log=False)

		factor = 1.15e-28
		pSFR = pUV * factor

		if log==True:
			pSFR = unp.log10(pSFR)

		return pSFR


class DoublePower:
	def __init__(self, phi_star, Mstar, alpha, beta):
		if phi_star < 0.:
			self.phi_star = 10**phi_star
		else:
			self.phi_star = phi_star
		self.Mstar = Mstar
		self.alpha = alpha
		self.beta = beta

	def get_phi_mag(self, MUV_array=None):
		if MUV_array==None:
			MUV_array = np.arange(-25.,-14.99,0.01)
		phi_mag = self.phi_star / ((10**(0.4*(self.alpha+1)*(MUV_array-self.Mstar))) + (10**(0.4*(self.beta+1)*(MUV_array-self.Mstar))))
		return MUV_array, unp.nominal_values(phi_mag), unp.std_devs(phi_mag)

	def get_puv(self, Mint=None, log=False):
		if Mint==None:
			Mint = -18.
		
		x, y, yerr = self.get_phi_mag()
		xlum = MUV_to_LUV(x)
		
		pUV_upp = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]+yerr[(x <= Mint)])
		pUV = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)])
		pUV_low = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]-yerr[(x <= Mint)])
		pUV_err = np.mean(abs(np.diff([pUV_upp, pUV, pUV_low])))

		if log==True:
			pUV = unp.log10(ufloat(pUV, pUV_err))
			pUV_err = unp.std_devs(pUV)
			pUV = unp.nominal_values(pUV)

		pUV = ufloat(pUV, pUV_err)
		return pUV

	def get_psfr(self, Mint=None, log=False):
		if Mint==None:
			Mint = -18.

		pUV = self.get_puv(Mint=Mint, log=False)

		factor = 1.15e-28

		pSFR = pUV * factor

		if log==True:
			pSFR = unp.log10(pSFR)

		return pSFR


class Mason15:
	def __init__(self):
		self.redshifts = np.array([0.,2.,4.,5.,6.,7.,8.,9.,10.,12.,14.,16])
		self.phi_star = 10**unp.uarray([-2.97,-2.52,-2.93,-3.12,-3.19,-3.48,-4.03,-4.50,-5.12,-5.94,-7.05,-8.25],[np.mean([0.07,0.08]),np.mean([0.07,0.09]),np.mean([0.13,0.19]),np.mean([0.15,0.24]),np.mean([0.16,0.25]),np.mean([0.18,0.32]),np.mean([0.26,0.72]),np.mean([0.29,1.36]),0.34,0.38,0.45,0.51])
		self.Mstar = unp.uarray([-19.9,-20.3,-21.2,-21.2,-20.9,-21.0,-21.3,-21.2,-21.1,-21.0,-20.9,-20.7],[0.1,0.1,0.2,0.2,0.2,0.2,0.4,0.4,0.5,0.5,0.5,0.6])
		self.alpha = unp.uarray([-1.68,-1.46,-1.64,-1.75,-1.83,-1.95,-2.10,-2.26,-2.47,-2.74,-3.11,-3.51],[0.09,0.09,0.11,0.13,0.15,0.17,0.20,0.22,0.26,0.30,0.38,0.46])

		print(f'Redshift range for Mason+15 UVLF is z~0.0-16.0.')

	def get_phi_mag(self, z=None, MUV_array=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		i = gf.find_nearest(self.redshifts, z)

		if MUV_array==None:
			MUV_array = np.arange(-25.,-14.99,0.01)
		phi_mag = (np.log(10.)/2.5) * self.phi_star[i] * ((10**(0.4*(self.Mstar[i]-MUV_array)))**(self.alpha[i]+1.)) * unp.exp(-10.**(0.4*(self.Mstar[i]-MUV_array)))
		return MUV_array, unp.nominal_values(phi_mag), unp.std_devs(phi_mag)
	
	def get_puv(self, z=None, Mint=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		if Mint==None:
			Mint = -18.
		
		if type(z) in [np.ndarray,list]:
			pUV_values, pUVerr_values = [], []
			for zi in z:
				x, y, yerr = self.get_phi_mag(z=zi)
				xlum = MUV_to_LUV(x)
		
				pUV_upp = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]+yerr[(x <= Mint)])
				pUV = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)])
				pUV_low = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]-yerr[(x <= Mint)])
				pUV_err = np.mean([pUV_upp-pUV, pUV-pUV_low], axis=0)

				pUV_values.append(pUV)
				pUVerr_values.append(pUV_err)
			return pUV_values, pUVerr_values

		else:
			x, y, yerr = self.get_phi_mag(z=z)
			xlum = MUV_to_LUV(x)
	
			pUV_upp = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]+yerr[(x <= Mint)])
			pUV = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)])
			pUV_low = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]-yerr[(x <= Mint)])
			pUV_err = np.mean([pUV_upp-pUV, pUV-pUV_low], axis=0)
			return pUV, pUV_err
	
	def get_psfr(self, z=None, Mint=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		if Mint==None:
			Mint = -18.

		pUV_val, pUV_err = self.get_puv(z=z, Mint=Mint)
		if type(pUV_val) in [np.ndarray,list]:
			pUV = unp.uarray(pUV_val,pUV_err)
		else:
			pUV = ufloat(pUV_val, pUV_err)

		factor = 1.15e-28
		pSFR = pUV * factor
		return unp.nominal_values(pSFR), unp.std_devs(pSFR)


class Bouwens15:
	def __init__(self):
		self.redshifts = np.array([3.8,4.9,5.9,6.8,7.9,10.4])
		self.phi_star = unp.uarray([1.97,0.74,0.50,0.29,0.21,0.008],[np.mean([0.34,0.29]),np.mean([0.18,0.14]),np.mean([0.22,0.16]),np.mean([0.21,0.12]),np.mean([0.23,0.11]),np.mean([0.004,0.003])])
		self.Mstar = unp.uarray([-20.88,-21.17,-20.94,-20.87,-20.63,-20.92],[0.08,0.12,0.20,0.26,0.36,0.00])
		self.alpha = unp.uarray([-1.64,-1.76,-1.87,-2.06,-2.02,-2.27],[0.04,0.05,0.10,0.13,0.23,0.00])

		print(len(self.redshifts), len(self.phi_star), len(self.Mstar), len(self.alpha))
		print(f'Redshift range for Bouwens+15 UVLF is z=3.8-10.4.')

	def get_phi_mag(self, z=None, MUV_array=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		i = gf.find_nearest(self.redshifts, z)

		if MUV_array==None:
			MUV_array = np.arange(-25.,-14.99,0.01)
		phi_mag = (np.log(10.)/2.5) * self.phi_star[i] * ((10**(0.4*(self.Mstar[i]-MUV_array)))**(self.alpha[i]+1.)) * unp.exp(-10.**(0.4*(self.Mstar[i]-MUV_array)))
		return MUV_array, unp.nominal_values(phi_mag), unp.std_devs(phi_mag)
	
	def get_puv(self, z=None, Mint=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		if Mint==None:
			Mint = -18.
		
		if type(z) in [np.ndarray,list]:
			pUV_values, pUVerr_values = [], []
			for zi in z:
				x, y, yerr = self.get_phi_mag(z=zi)
				xlum = MUV_to_LUV(x)
		
				pUV_upp = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]+yerr[(x <= Mint)])
				pUV = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)])
				pUV_low = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]-yerr[(x <= Mint)])
				pUV_err = np.mean([pUV_upp-pUV, pUV-pUV_low], axis=0)

				pUV_values.append(pUV)
				pUVerr_values.append(pUV_err)
			return pUV_values, pUVerr_values

		else:
			x, y, yerr = self.get_phi_mag(z=z)
			xlum = MUV_to_LUV(x)
	
			pUV_upp = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]+yerr[(x <= Mint)])
			pUV = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)])
			pUV_low = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]-yerr[(x <= Mint)])
			pUV_err = np.mean([pUV_upp-pUV, pUV-pUV_low], axis=0)
			return pUV, pUV_err
	
	def get_psfr(self, z=None, Mint=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		if Mint==None:
			Mint = -18.

		pUV_val, pUV_err = self.get_puv(z=z, Mint=Mint)
		if type(pUV_val) in [np.ndarray,list]:
			pUV = unp.uarray(pUV_val,pUV_err)
		else:
			pUV = ufloat(pUV_val, pUV_err)

		factor = 1.15e-28
		pSFR = pUV * factor
		return unp.nominal_values(pSFR), unp.std_devs(pSFR)


class Donnan23:
	def __init__(self):
		self.redshifts = np.array([8.0,9.0,10.5,13.25])
		self.phi_star = unp.uarray([3.30,2.10,3.32,0.51],[3.41,1.68,8.96,0.22])*(1.e-4)
		self.Mstar = unp.uarray([-20.02,-19.93,-19.12,-19.12],[0.55,0.58,1.68,0.00])
		self.alpha = unp.uarray([-2.04,-2.10,-2.10,-2.10],[0.29,0.00,0.00,0.00])
		self.beta = unp.uarray([-4.26,-4.29,-3.53,-3.53],[0.50,0.69,1.06,0.00])

		print(f'Redshift range for Donnan+23 UVLF is z~8.0-13.5.')

	def get_phi_mag(self, z=None, MUV_array=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		i = gf.find_nearest(self.redshifts, z)

		if MUV_array==None:
			MUV_array = np.arange(-25.,-14.99,0.01)
		phi_mag = self.phi_star[i] / ((10**(0.4*(self.alpha[i]+1)*(MUV_array-self.Mstar[i]))) + (10**(0.4*(self.beta[i]+1)*(MUV_array-self.Mstar[i]))))
		return MUV_array, unp.nominal_values(phi_mag), unp.std_devs(phi_mag)
	
	def get_puv(self, z=None, Mint=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		if Mint==None:
			Mint = -18.
		
		if type(z) in [np.ndarray,list]:
			pUV_values, pUVerr_values = [], []
			for zi in z:
				x, y, yerr = self.get_phi_mag(z=zi)
				xlum = MUV_to_LUV(x)
		
				pUV_upp = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]+yerr[(x <= Mint)])
				pUV = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)])
				pUV_low = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]-yerr[(x <= Mint)])
				pUV_err = np.mean([pUV_upp-pUV, pUV-pUV_low], axis=0)

				pUV_values.append(pUV)
				pUVerr_values.append(pUV_err)
			return pUV_values, pUVerr_values

		else:
			x, y, yerr = self.get_phi_mag(z=z)
			xlum = MUV_to_LUV(x)
	
			pUV_upp = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]+yerr[(x <= Mint)])
			pUV = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)])
			pUV_low = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]-yerr[(x <= Mint)])
			pUV_err = np.mean([pUV_upp-pUV, pUV-pUV_low], axis=0)
			return pUV, pUV_err
	
	def get_psfr(self, z=None, Mint=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		if Mint==None:
			Mint = -18.

		pUV_val, pUV_err = self.get_puv(z=z, Mint=Mint)
		if type(pUV_val) in [np.ndarray,list]:
			pUV = unp.uarray(pUV_val,pUV_err)
		else:
			pUV = ufloat(pUV_val, pUV_err)

		factor = 1.15e-28
		pSFR = pUV * factor
		return unp.nominal_values(pSFR), unp.std_devs(pSFR)



class Donnan24:
	def __init__(self):
		self.redshifts = np.array([9.0,10.0,11.0,12.5,14.5])
		self.phi_star = unp.uarray([39.2,16.4,9.86,0.99,0.28],[23.5,14.5,3.27,0.99,0.18])*(1.e-5)
		self.Mstar = unp.uarray([-19.70,-19.98,-20.73,-20.82,-20.82],[0.96,0.61,1.61,0.71,0.00])
		self.alpha = unp.uarray([-2.00,-1.98,-2.19,-2.19,-2.19],[0.47,0.40,0.69,0.00,0.00])
		self.beta = unp.uarray([-3.81,-4.05,-4.29,-4.29,-4.29],[0.49,0.00,1.30,0.00,0.00])
		print(f'Redshift range for Donnan+24 UVLF is z~9.0-14.5.')

	def get_phi_mag(self, z=None, MUV_array=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		i = gf.find_nearest(self.redshifts, z)

		if MUV_array==None:
			MUV_array = np.arange(-25.,-14.99,0.01)
		phi_mag = self.phi_star[i] / ((10**(0.4*(self.alpha[i]+1)*(MUV_array-self.Mstar[i]))) + (10**(0.4*(self.beta[i]+1)*(MUV_array-self.Mstar[i]))))
		return MUV_array, unp.nominal_values(phi_mag), unp.std_devs(phi_mag)
	
	def get_puv(self, z=None, Mint=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		if Mint==None:
			Mint = -18.
		
		if type(z) in [np.ndarray,list]:
			pUV_values, pUVerr_values = [], []
			for zi in z:
				x, y, yerr = self.get_phi_mag(z=zi)
				xlum = MUV_to_LUV(x)
		
				pUV_upp = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]+yerr[(x <= Mint)])
				pUV = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)])
				pUV_low = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]-yerr[(x <= Mint)])
				pUV_err = np.mean([pUV_upp-pUV, pUV-pUV_low], axis=0)

				pUV_values.append(pUV)
				pUVerr_values.append(pUV_err)
			return pUV_values, pUVerr_values

		else:
			x, y, yerr = self.get_phi_mag(z=z)
			xlum = MUV_to_LUV(x)
	
			pUV_upp = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]+yerr[(x <= Mint)])
			pUV = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)])
			pUV_low = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]-yerr[(x <= Mint)])
			pUV_err = np.mean([pUV_upp-pUV, pUV-pUV_low], axis=0)
			return pUV, pUV_err
	
	def get_psfr(self, z=None, Mint=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		if Mint==None:
			Mint = -18.

		pUV_val, pUV_err = self.get_puv(z=z, Mint=Mint)
		if type(pUV_val) in [np.ndarray,list]:
			pUV = unp.uarray(pUV_val,pUV_err)
		else:
			pUV = ufloat(pUV_val, pUV_err)

		factor = 1.15e-28
		pSFR = pUV * factor
		return unp.nominal_values(pSFR), unp.std_devs(pSFR)


class Harikane23_Schechter:
	def __init__(self):
		self.redshifts = np.array([9.,12.,16.])
		self.phi_star = unp.uarray([-4.83,-5.95,-5.84],[np.mean([0.49,0.37]),np.mean([0.18,1.84]),np.mean([0.47,4.03])])
		self.Mstar = unp.uarray([-21.24,-21.97,-20.80],[np.mean([0.45,0.59]),np.mean([0.11,2.88]),0.])
		self.alpha = unp.uarray([-2.35,-2.35,-2.35],[0.,0.,0.])

		print(len(self.redshifts), len(self.phi_star), len(self.Mstar), len(self.alpha))
		print(f'Redshift range for Harikane+23 UVLF is z~9.0-16.0.')

	def get_phi_mag(self, z=None, MUV_array=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		i = gf.find_nearest(self.redshifts, z)

		if MUV_array==None:
			MUV_array = np.arange(-25.,-14.99,0.01)
		phi_mag = (np.log(10.)/2.5) * self.phi_star[i] * ((10**(0.4*(self.Mstar[i]-MUV_array)))**(self.alpha[i]+1.)) * unp.exp(-10.**(0.4*(self.Mstar[i]-MUV_array)))
		return MUV_array, unp.nominal_values(phi_mag), unp.std_devs(phi_mag)
	
	def get_puv(self, z=None, Mint=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		if Mint==None:
			Mint = -18.
		
		if type(z) in [np.ndarray,list]:
			pUV_values, pUVerr_values = [], []
			for zi in z:
				x, y, yerr = self.get_phi_mag(z=zi)
				xlum = MUV_to_LUV(x)
		
				pUV_upp = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]+yerr[(x <= Mint)])
				pUV = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)])
				pUV_low = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]-yerr[(x <= Mint)])
				pUV_err = np.mean([pUV_upp-pUV, pUV-pUV_low], axis=0)

				pUV_values.append(pUV)
				pUVerr_values.append(pUV_err)
			return pUV_values, pUVerr_values

		else:
			x, y, yerr = self.get_phi_mag(z=z)
			xlum = MUV_to_LUV(x)
	
			pUV_upp = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]+yerr[(x <= Mint)])
			pUV = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)])
			pUV_low = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]-yerr[(x <= Mint)])
			pUV_err = np.mean([pUV_upp-pUV, pUV-pUV_low], axis=0)
			return pUV, pUV_err
	
	def get_psfr(self, z=None, Mint=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		if Mint==None:
			Mint = -18.

		pUV_val, pUV_err = self.get_puv(z=z, Mint=Mint)
		if type(pUV_val) in [np.ndarray,list]:
			pUV = unp.uarray(pUV_val,pUV_err)
		else:
			pUV = ufloat(pUV_val, pUV_err)

		factor = 1.15e-28
		pSFR = pUV * factor
		return unp.nominal_values(pSFR), unp.std_devs(pSFR)


class Harikane23_DPL:
	def __init__(self):
		self.redshifts = np.array([9.,12.,16.])
		self.phi_star = 10**unp.uarray([-3.5,-4.32,-4.71],[np.mean([0.65,1.53]),0.22,np.mean([0.33,2.83])])
		self.Mstar = unp.uarray([-19.33,-19.60,-19.60],[np.mean([2.24,0.96]),0.00,0.00])
		self.alpha = unp.uarray([-2.1,-2.1,-2.1],[0.,0.,0.])
		self.beta = unp.uarray([-3.27,-2.21,-2.70],[np.mean([0.34,0.37]),np.mean([1.07,1.06]),0.])
		print(f'Redshift range for Harikane+23 UVLF is z~9.0-16.0.')

	def get_phi_mag(self, z=None, MUV_array=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		i = gf.find_nearest(self.redshifts, z)

		if MUV_array==None:
			MUV_array = np.arange(-25.,-14.99,0.01)
		phi_mag = self.phi_star[i] / ((10**(0.4*(self.alpha[i]+1)*(MUV_array-self.Mstar[i]))) + (10**(0.4*(self.beta[i]+1)*(MUV_array-self.Mstar[i]))))
		return MUV_array, unp.nominal_values(phi_mag), unp.std_devs(phi_mag)
	
	def get_puv(self, z=None, Mint=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		if Mint==None:
			Mint = -18.
		
		if type(z) in [np.ndarray,list]:
			pUV_values, pUVerr_values = [], []
			for zi in z:
				x, y, yerr = self.get_phi_mag(z=zi)
				xlum = MUV_to_LUV(x)
		
				pUV_upp = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]+yerr[(x <= Mint)])
				pUV = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)])
				pUV_low = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]-yerr[(x <= Mint)])
				pUV_err = np.mean([pUV_upp-pUV, pUV-pUV_low], axis=0)

				pUV_values.append(pUV)
				pUVerr_values.append(pUV_err)
			return pUV_values, pUVerr_values

		else:
			x, y, yerr = self.get_phi_mag(z=z)
			xlum = MUV_to_LUV(x)
	
			pUV_upp = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]+yerr[(x <= Mint)])
			pUV = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)])
			pUV_low = trapezoid(xlum[(x <= Mint)], y[(x <= Mint)]-yerr[(x <= Mint)])
			pUV_err = np.mean([pUV_upp-pUV, pUV-pUV_low], axis=0)
			return pUV, pUV_err
	
	def get_psfr(self, z=None, Mint=None):
		assert z!=None, f'Please select a redshift to compute {list(self.redshifts)}'
		if Mint==None:
			Mint = -18.

		pUV_val, pUV_err = self.get_puv(z=z, Mint=Mint)
		if type(pUV_val) in [np.ndarray,list]:
			pUV = unp.uarray(pUV_val,pUV_err)
		else:
			pUV = ufloat(pUV_val, pUV_err)

		factor = 1.15e-28
		pSFR = pUV * factor
		return unp.nominal_values(pSFR), unp.std_devs(pSFR)