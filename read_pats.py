import numpy as np

def read_hfss_ffd(path):
    f = open(path, "r")

    # get theta discretization
    comps = f.readline().split()
    th = np.linspace(np.float64(comps[0]), np.float64(comps[1]), int(comps[2])) * np.pi / 180
    # get ph discretization
    comps = f.readline().split()
    ph = np.linspace(np.float64(comps[0]), np.float64(comps[1]), int(comps[2])) * np.pi / 180
    
    # get number of frequencies
    comps = f.readline().split()
    assert(comps[0] == "Frequencies")
    num_freqs = int(comps[1])

    ph, th = np.meshgrid(ph, th)
    Eth = np.zeros((num_freqs, np.shape(th)[0], np.shape(th)[1]), dtype=np.complex128) 
    Eph = np.zeros((num_freqs, np.shape(th)[0], np.shape(th)[1]), dtype=np.complex128)
    freqs = np.zeros((num_freqs))

    count = 0
    findex = -1
    while True:
        comps = f.readline().split()
        if len(comps) == 0:
            break
        if comps[0] == "Frequency":
            count = 0
            findex += 1
            freqs[findex] = np.float64(comps[1])
        else:
            i0 = count % np.shape(th)[1]
            i1 = int(count / np.shape(th)[1])

            Eth[findex, i1, i0] = np.float64(comps[0]) + 1j*np.float64(comps[1])
            Eph[findex, i1, i0] = np.float64(comps[2]) + 1j*np.float64(comps[3])
            count += 1

    f.close()

    return freqs, th, ph, Eth, Eph

def read_feko_ffe(path):
    f = open(path, "r")

    freqs = []
    th = []
    Eth = []
    num_th = 0
    ph = []
    Eph = []
    num_ph = 0

    count = 0

    Etht = []
    Epht = []

    while True:
        ln = f.readline()
        if ln == "":
            break
            
        comps = ln.split()
        if len(comps) == 0:
            continue

        if comps[0] == "#Frequency:":
            freqs.append(np.float64(comps[1]))

            if count != 0:
                count = 0
                Etht.append(Eth)
                Epht.append(Eph)

        elif comps[0] == "#Coordinate":
            assert(comps[2] == "Spherical")
        elif ln.startswith("#No. of Theta Samples"):
            num_th = int(comps[4])
        elif ln.startswith("#No. of Phi Samples"):
            num_ph = int(comps[4])
        elif not ln.startswith("#") and not ln.startswith("**"):
            # data line, create arrays
            if count == 0:
                th = np.zeros((num_th, num_ph))
                ph = np.zeros((num_th, num_ph))
                Eth = np.zeros((num_th, num_ph), dtype=np.complex128)
                Eph = np.zeros((num_th, num_ph), dtype=np.complex128)

            i1 = count % np.shape(th)[0]
            i0 = int(count / np.shape(th)[0])

            th[i1, i0] = np.float64(comps[0]) * np.pi / 180
            ph[i1, i0] = np.float64(comps[1]) * np.pi / 180

            Eth[i1, i0] = np.float64(comps[2]) + 1j*np.float64(comps[3])
            Eph[i1, i0] = np.float64(comps[4]) + 1j*np.float64(comps[5])

            count += 1

    Etht.append(Eth)
    Epht.append(Eph)

    Ethfull = np.array(Etht)
    Ephfull = np.array(Epht)

    f.close()

    return np.array(freqs), th, ph, Ethfull, Ephfull

def write_feko_ffe(path, freqs, th, ph, Eth, Eph):
    f = open(path, "w")
    f.write("##File Type: Far Field\n")
    f.write("##File Format: 8\n")
    f.write("##Source: helix_linear_pol_01\n")
    f.write("##Date: 2025-08-21 15:36:05\n")
    f.write("** File exported by Altair Feko - Solver (par) Version 2023.1-9308 from 2023-12-14\n\n\n")
    f.write("#Configuration Name: ElementConfiguration1\n")
    f.write("#Request Name: FarField1\n")
    for fi in range(len(freqs)):
        f.write(f"#Frequency:\t{freqs[fi]}\n")
        f.write("#Coordinate System: Spherical\n")
        f.write(f"#No. of Theta Samples: {np.size(th)}\n")
        f.write(f"#No. of Phi Samples: {np.size(ph)}\n")
        f.write(f"#Result Type: Directivity\n")
        f.write(f"#Efficiency: 1\n")
        f.write("#No. of Header Lines: 1\n")
        f.write('#       "Theta"             "Phi"           "Re(Etheta)"       "Im(Etheta)"        "Re(Ephi)"         "Im(Ephi)"   "Directivity(Theta)" "Directivity(Phi)" "Directivity(Total)\n')

        for pi in range(len(ph)):
            
            for ti in range(len(th)):
                f.write(f"\t{th[ti]*180/np.pi}\t{ph[pi]*180/np.pi}\t{np.real(Eth[fi,ti,pi])}\t{np.imag(Eth[fi,ti,pi])}\t{np.real(Eph[fi,ti,pi])}\t{np.imag(Eph[fi,ti,pi])}\t1\t1\t1\n") 

def get_nsi_nth_nph_nfreqs(path):
    f = open(path, "r", errors='ignore')

    # make sure first line has format we expect
    ln = f.readline()
    assert(ln.startswith("# Frequency Theta Phi dB(Etheta) Phase(Etheta) dB(Ephi) Phase(Ephi)"))

    th_unique = set()
    ph_unique = set()
    freq_unique = set()

    while True:
        ln = f.readline()
        if ln == "":
            break
        comps = ln.split()

        freq_unique.add(comps[0])
        th_unique.add(comps[2])
        ph_unique.add(comps[3])

    f.close()

    return len(th_unique), len(ph_unique), len(freq_unique)

def read_nsi_measurements(path, nth, nph, nfreqs):
    f = open(path, "r", errors='ignore')

    # make sure first line has format we expect
    ln = f.readline()
    assert(ln.startswith("# Frequency Theta Phi dB(Etheta) Phase(Etheta) dB(Ephi) Phase(Ephi)"))

    Eth = np.zeros((nfreqs, nth, nph), dtype=np.complex128) 
    Eph = np.zeros((nfreqs, nth, nph), dtype=np.complex128)
    th = np.zeros((nth, nph))
    ph = np.zeros((nth, nph))
    freqs = np.zeros((nfreqs))

    fi = 0
    ti = 0
    pi = 0

    while True:
        ln = f.readline()
        if ln == "":
            break
        comps = ln.split()
        
        scale_factor = 1
        if comps[1] == "GHz":
            scale_factor = 1e9
        else:
            assert(False)

        freqs[fi] = np.float64(comps[0]) * scale_factor
        if fi > 0:
            assert(th[ti, pi] == np.float64(comps[2]) * np.pi / 180.0)
            assert(ph[ti, pi] == np.float64(comps[3]) * np.pi / 180.0)


        th[ti, pi] = np.float64(comps[2]) * np.pi / 180.0
        ph[ti, pi] = np.float64(comps[3]) * np.pi / 180.0

        Eth[fi, ti, pi] = 10**(np.float64(comps[4])/20)*np.exp(1j*np.pi / 180 * np.float64(comps[5]))
        Eph[fi, ti, pi] = 10**(np.float64(comps[6])/20)*np.exp(1j*np.pi / 180 * np.float64(comps[7]))

        pi += 1
        if pi >= nph:
            pi = 0
            ti += 1
            if ti >= nth:
                pi = 0
                ti = 0
                fi += 1
    
    f.close()

    return freqs, th, ph, Eth, Eph