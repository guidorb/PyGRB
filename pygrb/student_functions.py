import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from astropy.io import ascii
import pickle
from astropy.coordinates import SkyCoord
import astropy.units as u

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

class load_jewels:
	def __init__(self, main_path=None):
		self.tab = ascii.read(main_path + 'borg_catalog.cat')
		self.spectra = pickle.load(open(main_path + 'borg_spectra.p', "rb"), encoding='latin1')
		self.coords = SkyCoord(self.tab['ra'], self.tab['dec'], unit=(u.deg,u.deg))

	def load_prism_spectrum(self, msaid):
		return (self.spectra[msaid]['lam'], self.spectra[msaid]['flux'], self.spectra[msaid]['err'])

	def get_object_info(self, msaid, print_entry=False):
		i = np.where(self.tab['msaid'] == msaid)[0]
		z = float(self.tab['z'][i][0])
		ra, dec = float(self.tab['ra'][i][0]), float(self.tab['dec'][i][0])
		
		if print_entry:
			print(f'MSAID={msaid}', f'z={z:.3f}', f'ra={ra:.5f}', f'dec={dec:.5f}')
		return (z, ra, dec)

	def plot_prism_spectrum(self, msaid, clim=(-0.04,0.09), ylims=None, xlims=None, plot_lines=True):
		# Cross-match the MSA ID with the catalog above
		i = np.where(self.tab['msaid'] == msaid)[0]

		if len(i) == 0:
			raise ValueError(f"MSA ID {msaid} not found in catalog.")

		z_spec = float(self.tab['z'][i][0])
		
		# Load the 1D and 2D spectra
		lam, flux, err = self.load_prism_spectrum(msaid)
		lam_A, lam0 = lam * 10000., (lam * 10000.) / (1.+z_spec)
		im = self.spectra[msaid]['2D']
		
		# Plot
		fig = plt.figure(figsize=(10.,6.))
		gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[0.25,0.75])
		ax1 = fig.add_subplot(gs[0])
		ax1.pcolormesh(lam, np.arange(np.shape(im)[0]), im, cmap='magma', clim=clim)
		ax1.set_ylim(0,np.shape(im)[0])
		ycent = int(np.shape(im)[0]/2)
		ax1.set_ylim(ycent-10,ycent+10)
		ax1.set_xlim(min(lam), max(lam))
		style_axes(ax1)
		ax1.set_xticklabels([])
		ax1.set_yticks([])
		ax1.set_yticklabels([])
		
		ax2 = fig.add_subplot(gs[1])
		ax2.step(lam, flux, color='C0', linewidth=1.25)
		ax2.set_xlim(ax1.set_xlim())
		
		ax2.plot(ax2.set_xlim(), [0.,0.], linestyle='--', color='black')
		style_axes(ax2, r'Observed Wavelength [$\mu$m]', r'Flux Density [$\mu$Jy]')
		
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
	
		ax2.annotate(xy=(0.025,0.9), xycoords=('axes fraction'), text=msaid+r', $z_{\rm spec.}=%0.3f$'%(z_spec), fontsize=15)
		
		midpoint = ax2.set_ylim()[0] + ((ax2.set_ylim()[1] - ax2.set_ylim()[0]) * 0.6)
		highpoint = ax2.set_ylim()[0] + ((ax2.set_ylim()[1] - ax2.set_ylim()[0]) * 0.7)
		textpoint = ax2.set_ylim()[0] + ((ax2.set_ylim()[1] - ax2.set_ylim()[0]) * 0.725)
		if plot_lines==True:
			lines = {r'Ly$\alpha$':[1215.67],
					 'C IV':[np.mean([1548.187,1550.772])],
					 r'He II + O III]':[1640.42,np.mean([1660.809,1666.150])],
					 r'C III]':[1908.734],
					 # 'Mg II]':[2799.1165],
					 # 'He I UV':[2945.106],
					 '[O II]':[3728.4835000000003],
					 '[Ne III]':[3968.59, 3869.86],
					 r'H$\delta$':[4102.8922],
					 r'H$\gamma$ + [O III]':[4353.05985],
					 r'He I_1':[4472.734],
					 r'He II':[4687.015],
					 r'H$\beta$':[4862.6830],
					 '[O III]':[4960.295,5008.240],
					 'He I_2':[5877.252],
					 '[O I]':[6302.046],
					r'H$\alpha$':[6564.608],
					 '[S II]':[6725.4845000000005],
					 'He I_3':[7065.196]
					 }
			
			for line in lines:
				for l in lines[line]:
					lz = (l / 10000.) * (1.+z_spec)
					if (lz<ax2.set_xlim()[0]) | (lz>ax2.set_xlim()[1]):
						continue
					if line==r'H$\beta$':
						ax2.plot([lz,lz], [midpoint,highpoint], linestyle='--', color='darkgray')
					elif line=='[O III]':
						ax2.plot([lz,lz], [midpoint,highpoint], linestyle='--', color='darkgray')
					else:
						ax2.plot([lz,lz], [midpoint,highpoint], linestyle='--', color='darkgray')
	
				if (lz<ax2.set_xlim()[0]) | (lz>ax2.set_xlim()[1]):
					continue
	
				label = line.split('_')[0]
				if label=='[Ne III]':
					ax2.text(lz+0.05, textpoint, va='bottom', ha='center', color='darkgray', s=label, rotation=90, fontsize=10.)
				elif label=='[O III]':
					ax2.text(lz+0.08, textpoint, va='bottom', ha='center', color='darkgray', s=label, rotation=90, fontsize=10.)
				elif label==r'H$\beta$':
					ax2.text(lz, textpoint, va='bottom', ha='center', color='darkgray', s=label, rotation=90, fontsize=10.)
				elif label==r'H$\alpha$':
					ax2.text(lz-0.065, textpoint, va='bottom', ha='center', color='darkgray', s=label, rotation=90, fontsize=10.)
				else:
					ax2.text(lz, textpoint, va='bottom', ha='center', color='darkgray', s=label, rotation=90, fontsize=10.)
	
		plt.tight_layout()
		plt.show()	