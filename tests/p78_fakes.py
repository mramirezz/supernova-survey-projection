import numpy as np
from pipeline78.store import save_template

def sed(w, T=10000.0):
    bb = 1.0 / (w**5 * (np.exp(1.4388e8 / (w * T)) - 1.0))
    return bb / bb.max() * 0.5

def fake_template(d, sn="FAKE1", clase="Ia", t_peak=55000.0, n_ep=120):
    w = np.arange(3005.0, 9195.0, 1.0)
    t = t_peak + np.arange(-20.0, n_ep - 20.0, 1.0)
    prof = np.exp(-0.5 * ((t - t_peak) / 15.0) ** 2) + 0.05
    save_template(d, t, w, prof[:, None] * sed(w)[None, :],
                  dict(sn=sn, clase=clase, md5_src="fake", n_epochs=int(t.size), t_first=float(t[0]),
                       t_last=float(t[-1]), wmin=float(w[0]), wmax=float(w[-1]), n_neg=0))

def write_dat(path, times, wave, fluxes):
    with open(path, "w") as fh:
        for t, f in zip(times, fluxes):
            fh.write(f"# time:\t{t}\n# SPEC\n#      WAVE   FLUX\n")
            for wi, fi in zip(wave, f):
                fh.write(f"{wi} {fi}\n")
