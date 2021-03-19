# this file creates the 16384x16384x10 grid for the mass


from astropy.table import Table
import numpy as np
import numba
import matplotlib.pyplot as plt

# the different columns in our dataset:
# ra_gal: galaxy right ascension (degrees)
# dec_gal: galaxy declination (degrees)
# ra_gal_mag: magnified galaxy right ascension (degrees)
# dec_gal_mag: magnified galaxy declination (degrees)
# kappa: convergence
# gamma1: shear
# gamma2: shear
# z_cgal: galaxy true redshift
# z_cgal_v: galaxy observed redshift
# unique_gal_id: unique galaxy id
# lmstellar: logarithm of the stellar mass
columns = ['ra_gal', 'dec_gal', 'ra_gal_mag', 'dec_gal_mag', 'kappa', 'gamma1',
           'gamma2', 'z_cgal', 'z_cgal_v', 'unique_gal_id', 'lmstellar']

# size is the size of one axis of the map, in this case 16384
size = 2 ** 14


def get_map(index):
    # we create an empty map filled with zeros, to which we add the correct values
    # when going through the dataset
    map = np.zeros((size, size, 10), dtype=np.int32)-1
    val_map = np.zeros((size, size, 10), dtype=np.float32)

    # this @numba.njit translates the function directly into machine code for a considerable speedup
    @numba.njit
    def map_to_place(ra_index, dec_index, z_index, data, start, map, val_map):
        for j in np.arange(len(data)):
            # we select indices from the arrays
            x, y, z = ra_index[j], dec_index[j], z_index[j]

            # we use these indices to assign the mass to its corresponding position
            if 0 <= x < size and 0 <= y < size and 0 <= z < 10:
                if val_map[x, y, z] < data[i, 10]:
                    map[x, y, z] = j + start
                    val_map[x, y, z] = data[i, 10]

        return map, val_map

    def data_to_map(data, start, map, val_map):
        # we filter out all the entries that do not fall within our scope,
        # there are some entries in the dataset with -270 degrees,
        # that shouldn't be in there
        # between = (0 < data[:, 2]) * (data[:, 2] < 90) * (0 < data[:, 3]) * (data[:, 3] < 90) * (
        #             0.07296 < data[:, 7]) * (data[:, 7] < 1.41708)
        # data = data[np.where(between)]
        # print(len(data))

        # get the indexes by redistributing it from 0-90 to 0-16383
        ra_index = (data[:, 2] / 90 * size).astype(np.int32)
        dec_index = (data[:, 3] / 90 * size).astype(np.int32)
        d_z = 1.41708 - 0.07296

        # get the indexes by redistributing it from 0.07296-1.41708 to 0-9
        z_index = ((data[:, 7] - .07296) / d_z * 10).astype(np.int32)

        # using these indices we can fill in our temp_map
        return map_to_place(ra_index, dec_index, z_index, data, start, map, val_map)

    # here we load the MICECATv2.0 dataset. Using memmap avoids out of memory errors
    dat = Table.read("C:/datasets/8336.fits", format='fits', memmap=True)

    # we load the data by batch to avoid memory errors
    batch_size = 50_000_000
    for i in range(0, 500_000_000, batch_size):
        print(i)
        # we select a piece of the file and process it using the above functions
        file = dat[i:i+batch_size].to_pandas().to_numpy()
        map, val_map = data_to_map(file, i, map, val_map)

    # the map is saved in the right directory
    np.save(f"C:/datasets/{size}x{size}x10_index_{columns[index]}.npy", map)

    # here we show the map to make sure everything went according to plan
    plt.imshow(map.mean(2))
    plt.show()


get_map(10)