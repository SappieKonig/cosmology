from astropy.table import Table
import numpy as np


dat = Table.read("C:/datasets/8336.fits", format='fits', memmap=True)

mass = [dat[i:i+50_000_000].to_pandas().to_numpy()[:, [2, 3, 7, 4, 10]] for i in range(0, 500_000_000, 50_000_000)]
x = dat[:5_000_000].to_pandas()
print(x['ra_gal_mag'].dtype, x['dec_gal_mag'].dtype, x['z_cgal'].dtype, x['lmstellar'].dtype)
print([i.shape for i in mass])
mass = np.concatenate(mass)
print(mass.dtype, mass.shape)
print(mass[:, -2])
np.save("C:/datasets/pos_kappa_mass_by_index", mass)
