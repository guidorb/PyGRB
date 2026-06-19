"""
Helper functions for working with NIRSpec PRISM spectra from DJA v4.4

This module provides convenient functions to:
- Load and filter the NIRSpec catalog
- Access spectra from the combined FITS table
- Plot and analyze spectra
- Work with emission line measurements

Author: Based on DJA documentation
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.utils.data import download_file
from grizli import utils
import os
import warnings
warnings.filterwarnings('ignore')


class NIRSpecCatalog:
    """
    A class to work with the NIRSpec merged catalog v4.4
    """
    
    def __init__(self, version="v4.4", cache=True, local_dir=None, download=False):
        """
        Initialize and load the NIRSpec catalog
        
        Parameters
        ----------
        version : str
            Catalog version (default: "v4.4")
        cache : bool
            Whether to cache downloads (default: True)
        local_dir : str, optional
            Path to local directory with downloaded files.
            If provided, files will be loaded from here instead of downloading.
            Example: "/Users/guidorb/Dropbox/Catalogs/DJA/NIRSpec/"
        """
        self.version = version
        self.cache = cache
        self.local_dir = local_dir
        self.url_prefix = "https://s3.amazonaws.com/msaexp-nirspec/extractions"
        
        # Load catalog
        print(f"Loading NIRSpec catalog {version}...")
        
        if local_dir is not None:
            # Load from local directory
            catalog_file = os.path.join(local_dir, f"dja_msaexp_emission_lines_{version}.csv.gz")
            if not os.path.exists(catalog_file):
                catalog_file = os.path.join(local_dir, f"dja_msaexp_emission_lines_{version}.csv")
            
            if not os.path.exists(catalog_file):
                raise FileNotFoundError(f"Catalog file not found: {catalog_file}")
            
            print(f"Loading from: {catalog_file}")
            self.catalog = utils.read_catalog(catalog_file, format='csv')
        elif local_dir is None and download is False:
            # Load from local directory
            self.local_dir = f"/Users/guidorb/Dropbox/Catalogs/DJA/NIRSpec/"

            catalog_file = f"{self.local_dir}dja_msaexp_emission_lines_{version}.csv"

            if not os.path.exists(catalog_file):
                raise FileNotFoundError(f"Catalog file not found: {catalog_file}")
            
            print(f"Loading from: {catalog_file}")
            self.catalog = utils.read_catalog(catalog_file, format='csv')
        elif local_dir is None and download is True:

            self.local_dir = os.getcwd() + '/'

            # Download from web
            table_url = f"{self.url_prefix}/dja_msaexp_emission_lines_{version}.csv.gz"
            self.catalog = utils.read_catalog(
                download_file(table_url, cache=cache), 
                format='csv'
            )
        
        print(f"Loaded {len(self.catalog)} entries")
        
        # Generate full MSA IDs from filenames and add to catalog
        # Format: "{prog_id}_{msaid}" (e.g., "5224_131711")
        print("Generating MSA IDs from filenames...")
        msaid_full = self._extract_msaids_from_filenames(self.catalog['file'])
        
        # Add as new column (msaid_full to distinguish from existing msaid column)
        self.catalog['msaid_full'] = msaid_full
        print(f"Added 'msaid_full' column to catalog")
        # print(f"  Example: {msaid_full[0]} from {self.catalog['file'][0]}")
        
        # Spectra tables (loaded on demand)
        self.prism_spectra = None
        self.grating_spectra = {}
        
        # MSA ID lookup dictionaries (created when spectra are loaded)
        self.prism_msaid_index = None
        self.grating_msaid_index = {}
        
    def filter(self, grating=None, grade=None, z_min=None, z_max=None, 
               root=None, custom_mask=None):
        """
        Filter the catalog based on various criteria
        
        Parameters
        ----------
        grating : str, optional
            Grating name (e.g., 'PRISM', 'G140M', 'G235M', 'G395M')
        grade : int or list, optional
            Redshift grade (3=robust, 2=ambiguous, 1=no features, 0=DQ issues)
        z_min : float, optional
            Minimum redshift
        z_max : float, optional
            Maximum redshift
        root : str, optional
            Root name (survey/field identifier)
        custom_mask : array-like, optional
            Custom boolean mask for selection
            
        Returns
        -------
        filtered_catalog : Table
            Filtered catalog entries
        """
        mask = np.ones(len(self.catalog), dtype=bool)
        
        if grating is not None:
            mask &= self.catalog['grating'] == grating
            
        if grade is not None:
            if isinstance(grade, (list, tuple)):
                grade_mask = np.zeros(len(self.catalog), dtype=bool)
                for g in grade:
                    grade_mask |= self.catalog['grade'] == g
                mask &= grade_mask
            else:
                mask &= self.catalog['grade'] == grade
                
        if z_min is not None:
            mask &= self.catalog['zrf'] >= z_min
            
        if z_max is not None:
            mask &= self.catalog['zrf'] <= z_max
            
        if root is not None:
            mask &= np.array([r.startswith(root) for r in self.catalog['root']])
            
        if custom_mask is not None:
            mask &= custom_mask
            
        print(f"Filter results: {mask.sum()} / {len(self.catalog)} entries selected")
        return self.catalog[mask]
    
    @staticmethod
    def _extract_msaid_from_filename(filename):
        """
        Extract MSA ID from a single filename.
        
        For example:
          'mom-uds02-v4_prism-clear_5224_131711.spec.fits' -> '5224_131711'
          'abell2744-castellano1-v4_prism-clear_3073_21088.spec.fits' -> '3073_21088'
        
        Parameters
        ----------
        filename : str
            Spectrum filename
            
        Returns
        -------
        msaid : str
            MSA ID in format "{prog_id}_{msaid}"
        """
        # Remove the .spec.fits extension
        base = filename.replace('.spec.fits', '')
        
        # Split by underscore and get last two parts
        parts = base.split('_')
        msaid = f"{parts[-2]}_{parts[-1]}"
        
        return msaid
    
    @staticmethod
    def _extract_msaids_from_filenames(filenames):
        """
        Extract MSA IDs from multiple filenames.
        
        Parameters
        ----------
        filenames : array-like
            Array of spectrum filenames
            
        Returns
        -------
        msaids : np.array
            Array of MSA IDs
        """
        return np.array([NIRSpecCatalog._extract_msaid_from_filename(f) for f in filenames])
    
    def load_prism_spectra(self):
        """
        Load the combined PRISM spectra FITS table
        
        Returns
        -------
        prism_spectra : Table
            Combined PRISM spectra table
        """
        if self.prism_spectra is not None:
            print("PRISM spectra already loaded")
            return self.prism_spectra
            
        print("Loading combined PRISM spectra...")
        
        filename = f"dja_msaexp_emission_lines_{self.version}.prism_spectra.fits"

        filepath = os.path.join(self.local_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PRISM spectra file not found: {filepath}")
        else:
            print(f"Loading from: {filepath}")
            self.prism_spectra = utils.read_catalog(filepath, format='fits')

        # else:
        #     # Download from web
        #     print("(~595 MB file, may take a few minutes on first download)")
        #     url = f"{self.url_prefix}/{filename}"
        #     self.prism_spectra = utils.read_catalog(
        #         download_file(url, cache=self.cache),
        #         format='fits'
        #     )
        
        print(f"Loaded {np.shape(self.prism_spectra['flux'])[1]} PRISM spectra")
        
        # Create MSA ID index for fast lookup
        print("Creating MSA ID index for PRISM spectra...")
        prism_mask = self.catalog['grating'] == 'PRISM'
        prism_catalog = self.catalog[prism_mask]
        
        self.prism_msaid_index = {}
        for i, msaid_full in enumerate(prism_catalog['msaid_full']):
            if msaid_full in self.prism_msaid_index:
                # Handle duplicates (multiple observations of same object)
                if not isinstance(self.prism_msaid_index[msaid_full], list):
                    self.prism_msaid_index[msaid_full] = [self.prism_msaid_index[msaid_full]]
                self.prism_msaid_index[msaid_full].append(i)
            else:
                self.prism_msaid_index[msaid_full] = i
        
        print(f"MSA ID index created with {len(self.prism_msaid_index)} unique MSA IDs")
        # return self.prism_spectra
    
    def load_grating_spectra(self, grating, filt):
        """
        Load combined grating spectra FITS table
        
        Parameters
        ----------
        grating : str
            Grating name ('G140M', 'G140H', 'G235M', 'G235H', 'G395M', 'G395H')
        filt : str
            Filter name ('F070LP', 'F100LP', 'F170LP', 'F290LP')
            
        Returns
        -------
        grating_spectra : Table
            Combined grating spectra table
        """
        key = (grating, filt)
        
        if key in self.grating_spectra:
            print(f"{grating}-{filt} spectra already loaded")
            return self.grating_spectra[key]
            
        print(f"Loading combined {grating}-{filt} spectra...")
        
        filename = f"dja_msaexp_emission_lines_{self.version}.{grating}-{filt}_spectra.fits".lower()
        
        if self.local_dir is not None:
            # Load from local directory
            filepath = os.path.join(self.local_dir, filename)
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Grating spectra file not found: {filepath}")
            
            print(f"Loading from: {filepath}")
            self.grating_spectra[key] = utils.read_catalog(filepath, format='fits')
        else:
            # Download from web
            url = f"{self.url_prefix}/{filename}"
            self.grating_spectra[key] = utils.read_catalog(
                download_file(url, cache=self.cache),
                format='fits'
            )
        
        print(f"Loaded {len(self.grating_spectra[key])} {grating}-{filt} spectra")
        return self.grating_spectra[key]
    
    def get_spectrum_by_msaid(self, msaid, grating='PRISM', index=0, verbose=False):
        """
        Get spectrum data by MSA ID
        
        Parameters
        ----------
        msaid : str
            MSA ID in format "{prog_id}_{msaid}" (e.g., "5224_131711")
        grating : str
            Grating name (default: 'PRISM')
        index : int
            If multiple observations exist for this MSA ID, which one to return (default: 0)
            
        Returns
        -------
        spectrum_data : dict
            Dictionary with wavelength, flux, error, valid mask, and metadata
        """
        if grating == 'PRISM':
            if self.prism_spectra is None:
                self.load_prism_spectra()
            
            if msaid not in self.prism_msaid_index:
                raise ValueError(f"MSA ID '{msaid}' not found in PRISM catalog")
            
            # Get the column index (or indices if multiple observations)
            idx = self.prism_msaid_index[msaid]
            
            if isinstance(idx, list):
                if index >= len(idx):
                    raise ValueError(f"Index {index} out of range. MSA ID '{msaid}' has {len(idx)} observations.")
                if verbose==True:
                    print(f"Found {len(idx)} observations for MSA ID '{msaid}', using observation {index}")
                idx = idx[index]
            
            # Get catalog entry
            prism_mask = self.catalog['grating'] == 'PRISM'
            prism_catalog = self.catalog[prism_mask]
            catalog_entry = prism_catalog[idx]
            
            if verbose==True:
                print(f"Found: {catalog_entry['file']}")
                print(f"  Redshift: z = {catalog_entry['zrf']:.4f}")
                print(f"  Grade: {catalog_entry['grade']}")
            
            # Extract spectrum (idx is the column index)
            wave = self.prism_spectra['wave']
            flux = self.prism_spectra['flux'][:, idx]
            err = self.prism_spectra['err'][:, idx]
            full_err = self.prism_spectra['full_err'][:, idx]
            valid = self.prism_spectra['valid'][:, idx]
            
            return {
                'wave': wave,
                'flux': flux,
                'err': err,
                'full_err': full_err,
                'valid': valid,
                'catalog_entry': catalog_entry,
                'wave_valid': wave[valid],
                'flux_valid': flux[valid],
                'err_valid': err[valid],
                'full_err_valid': full_err[valid]
            }
        else:
            raise NotImplementedError(f"MSA ID lookup for {grating} not yet implemented")
    
    def get_spectrum(self, filename, from_table=True):
        """
        Get spectrum data for a specific file
        
        Parameters
        ----------
        filename : str
            Spectrum filename (from catalog 'file' column)
        from_table : bool
            If True, use combined table; if False, download individual file
            
        Returns
        -------
        spectrum_data : dict
            Dictionary with wavelength, flux, error, and metadata
        """
        # Get catalog entry
        cat_entry = self.catalog[self.catalog['file'] == filename]
        if len(cat_entry) == 0:
            raise ValueError(f"File {filename} not found in catalog")
        cat_entry = cat_entry[0]
        
        if from_table:
            # Determine which table to use
            grating = cat_entry['grating']
            
            if grating == 'PRISM':
                if self.prism_spectra is None:
                    self.load_prism_spectra()
                spec_table = self.prism_spectra
            else:
                filt = cat_entry['filter']
                key = (grating, filt)
                if key not in self.grating_spectra:
                    self.load_grating_spectra(grating, filt)
                spec_table = self.grating_spectra[key]
            
            # Find spectrum in table
            match = spec_table['file'] == filename
            if match.sum() == 0:
                raise ValueError(f"Spectrum {filename} not found in combined table")
            
            spec_data = spec_table[match][0]
            
            return {
                'wave': spec_data['wave'],
                'flux': spec_data['flux'],
                'err': spec_data['err'],
                'full_err': spec_data['full_err'],
                'valid': spec_data['valid'],
                'R': spec_data['R'],
                'to_flam': spec_data['to_flam'],
                'catalog_entry': cat_entry
            }
        else:
            # Download individual file (requires msaexp)
            import msaexp.spectrum
            url = f"{self.url_prefix}/{cat_entry['root']}/{filename}"
            spec = msaexp.spectrum.SpectrumSampler(url)
            
            return {
                'spec_obj': spec,
                'wave': spec.spec_wobs,
                'flux': spec.spec['flux'],
                'err': spec.spec['err'],
                'full_err': spec.spec['full_err'],
                'valid': spec.valid,
                'R': spec.spec['R'],
                'to_flam': spec.spec['to_flam'],
                'catalog_entry': cat_entry
            }
    
    def plot_spectrum(self, filename, show_err=True, xlim=None, ylim=None, 
                     show_lines=False, z=None, figsize=(12, 4)):
        """
        Quick plot of a spectrum
        
        Parameters
        ----------
        filename : str
            Spectrum filename
        show_err : bool
            Show error spectrum
        xlim : tuple, optional
            Wavelength limits (microns)
        ylim : tuple, optional
            Flux limits
        show_lines : bool
            Overplot common emission lines at redshift z
        z : float, optional
            Redshift for line positions (uses catalog redshift if None)
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        # Get spectrum
        spec = self.get_spectrum(filename)
        
        if z is None:
            z = spec['catalog_entry']['zrf']
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot spectrum
        valid = spec['valid']
        ax.plot(spec['wave'][valid], spec['flux'][valid], 
                'k-', linewidth=0.8, label='Spectrum')
        
        if show_err:
            ax.plot(spec['wave'][valid], spec['err'][valid], 
                    'r-', linewidth=0.4, alpha=0.3, label='Error')
        
        # Add emission lines if requested
        if show_lines and z > 0:
            from grizli import utils as grizli_utils
            line_wavelengths, line_ratios = grizli_utils.get_line_wavelengths()
            
            common_lines = ['Lya', 'OII', 'Hb', 'OIII-4959', 'OIII-5007', 'Ha']
            
            ymin, ymax = ax.get_ylim()
            for line_name in common_lines:
                if line_name in line_wavelengths:
                    line_wave_rest = line_wavelengths[line_name][0] / 1e4  # Convert to microns
                    line_wave_obs = line_wave_rest * (1 + z)
                    
                    if xlim is None or (xlim[0] <= line_wave_obs <= xlim[1]):
                        ax.axvline(line_wave_obs, color='blue', 
                                 linestyle='--', alpha=0.3, linewidth=1)
                        ax.text(line_wave_obs, ymax * 0.95, line_name, 
                               rotation=90, va='top', fontsize=8, color='blue', alpha=0.7)
        
        ax.set_xlabel('Observed Wavelength [μm]', fontsize=12)
        ax.set_ylabel('Flux [μJy]', fontsize=12)
        ax.set_title(f"{filename} (z={z:.3f})", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        
        plt.tight_layout()
        return fig, ax
    
    def get_statistics(self):
        """
        Print summary statistics of the catalog
        """
        print("\n" + "="*70)
        print("NIRSpec Catalog Statistics")
        print("="*70)
        
        print(f"\nTotal entries: {len(self.catalog)}")
        
        # By grating
        print("\nBy Grating:")
        for grating in np.unique(self.catalog['grating']):
            count = (self.catalog['grating'] == grating).sum()
            print(f"  {grating}: {count}")
        
        # By grade
        print("\nBy Grade:")
        for grade in sorted(np.unique(self.catalog['grade'])):
            count = (self.catalog['grade'] == grade).sum()
            grade_desc = {3: 'Robust', 2: 'Ambiguous', 1: 'No features', 
                         0: 'DQ issues', -1: 'Not graded'}
            desc = grade_desc.get(grade, 'Unknown')
            print(f"  Grade {grade} ({desc}): {count}")
        
        # Redshift distribution
        has_z = self.catalog['zrf'] > 0
        print(f"\nRedshift range (grade >= 2): {self.catalog['zrf'][has_z & (self.catalog['grade'] >= 2)].min():.3f} - {self.catalog['zrf'][has_z & (self.catalog['grade'] >= 2)].max():.3f}")
        
        print("="*70 + "\n")


# # Example usage
# if __name__ == "__main__":
#     # Initialize catalog with local directory (or None to download)
#     local_dir = "/Users/guidorb/Dropbox/Catalogs/DJA/NIRSpec/"
    
#     # Set to None to download instead
#     # local_dir = None
    
#     cat = NIRSpecCatalog(local_dir=local_dir)
    
#     # Get statistics
#     cat.get_statistics()
    
#     # Filter for PRISM grade 3
#     prism_g3 = cat.filter(grating='PRISM', grade=3)
    
#     # Load spectra
#     cat.load_prism_spectra()
    
#     # Plot an example
#     if len(prism_g3) > 0:
#         example_file = prism_g3['file'][0]
#         fig, ax = cat.plot_spectrum(example_file, show_lines=True)
#         plt.savefig('/mnt/user-data/outputs/example_spectrum_helper.png', dpi=150)
#         print(f"\nExample plot saved for {example_file}")
#         plt.close()