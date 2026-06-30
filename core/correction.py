# 📚 IMPORTS PARA CORRECTION
# ===========================
import os
import numpy as np
import pandas as pd
from scipy import interpolate
from .utils import DL_calculator


def sample_host_extinction_mixture(n_samples=1, tau=0.4, Av_max=3.0, Rv=3.1,
                                   frac_zero=0.4, sigma_zero=0.01, random_state=None):
    """
    Simula el enrojecimiento del host (E(B–V)) usando un modelo mixto:
    una fracción de supernovas sin extinción (E(B–V) ≈ 0) y otra fracción con
    extinción muestreada desde una distribución exponencial en A_V.

    Justificación académica:
    ------------------------
    - Las distribuciones observadas de extinción en galaxias anfitrionas muestran
      un exceso de supernovas con E(B–V) ≈ 0, especialmente en galaxias elípticas
      o de baja masa (Hallgren et al. 2023, ApJ 949, 76; Holwerda et al. 2015).
    - Simulaciones cosmológicas modernas y estudios de curvas de luz (e.g., Jha et al. 2007;
      Kessler et al. 2009; Brout & Scolnic 2021) utilizan modelos de mezcla:
        - Gauss estrecha centrada en 0 + cola exponencial para polvo.
    - Escala típica para distribución exponencial en A_V: τ ≈ 0.4 mag
      (Holwerda et al. 2015, MNRAS 449, 4277).
    - Relación R_V ≈ 3.1 es estándar (Cardelli et al. 1989).
    - σ ≈ 0.01–0.03 mag en la componente gaussiana es consistente con
      los errores de color intrínseco (Jha et al. 2007; Scolnic et al. 2021, Pantheon+).

    Parámetros:
    -----------
    n_samples : int
        Número de supernovas a simular.
    tau : float
        Parámetro de escala de la distribución exponencial de A_V (mag).
    Av_max : float
        Corte máximo para A_V (evita valores físicamente extremos).
    Rv : float
        Relación entre A_V y E(B–V): A_V = R_V * E(B–V).
    frac_zero : float
        Fracción de eventos sin polvo (E(B–V) ≈ 0), típicamente 30%–50%.
    sigma_zero : float
        Dispersión (en mag) de la componente gaussiana centrada en 0.
    random_state : int, opcional
        Semilla para reproducibilidad.

    Retorna:
    --------
    ebmv_host : ndarray
        Arreglo de valores de E(B–V)_host simulados.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # FIX 2026-06-28: muestreo por-evento con un Bernoulli, NO un conteo entero.
    # Antes era `n_zero = int(frac_zero * n_samples)`, que con n_samples=1 (como la
    # proyección llama, una SN a la vez) da int(0.4)=0 -> la fracción sin-polvo NUNCA
    # aplicaba (toda SN salía con polvo). Con el Bernoulli, cada evento tiene
    # probabilidad frac_zero de ser limpio y la fracción correcta emerge sobre la
    # población. Verificado en aislado: da 40/20/20 con n_samples=1 (ver bitácora).
    is_zero = np.random.random(n_samples) < frac_zero
    ebmv_host = np.empty(n_samples)

    # Componente SIN polvo (Gauss estrecha centrada en 0)
    n_zero = int(is_zero.sum())
    ebmv_host[is_zero] = np.abs(np.random.normal(0, sigma_zero, size=n_zero))

    # Componente CON polvo (exponencial en A_V, truncada en Av_max, convertida a E(B-V))
    Av_samples = np.clip(np.random.exponential(tau, size=n_samples - n_zero), 0, Av_max)
    ebmv_host[~is_zero] = Av_samples / Rv

    return ebmv_host


def sample_extinction_by_type(sn_type="Ia", n_samples=1, random_state=None):
    """
    Muestrea E(B-V) de host por tipo de SN. Lee los parámetros del modelo de mezcla
    desde config.EXTINCTION_CONFIG (FUENTE ÚNICA DE VERDAD: valores y citas allí) y
    delega en sample_host_extinction_mixture().

    Justificación de los valores (resumen; detalle y citas en config.EXTINCTION_CONFIG):
    - Modelo de MEZCLA (frac_zero sin polvo + cola exponencial en A_V): estándar en
      simulaciones de SNe (Kessler+2009; Brout & Scolnic 2021). La distribución de
      core-collapse sigue a Hatano+1998 como en los frameworks modernos de simulación
      de surveys (Vincenzi+2019, usado en DES/LSST).
    - Ia : tau=0.35 (Holwerda+2015), frac_zero=0.40 (Ia en poblaciones limpias+polvorientas; Holwerda+2015, Brout&Scolnic+2021).
    - II : tau=0.25, frac_zero=0.20 (de Jaeger+2018: el reddening de host NO es dominante en II).
    - Ibc: tau=0.50 (Stritzinger+2018, CSP-I SE SNe <A_V>~0.5). II < Ibc por ~2x
           (Prentice 2016 vía Vincenzi+2019: Ib/Ic 2-3x mas extinguidas que II).

    Parámetros
    ----------
    sn_type : str        'Ia', 'II', 'Ib', 'Ic', 'Ibc'
    n_samples : int
    random_state : int, opcional   Semilla para reproducibilidad.

    Retorna
    -------
    ebmv_host : ndarray   E(B-V) de host muestreado.
    """
    from config import EXTINCTION_CONFIG

    key = {"IA": "SNIa", "II": "SNII",
           "IB": "SNIbc", "IC": "SNIbc", "IBC": "SNIbc"}.get(sn_type.upper())
    if key is None:
        raise ValueError(f"Tipo de supernova no reconocido: {sn_type}")

    p = EXTINCTION_CONFIG[key]
    return sample_host_extinction_mixture(
        n_samples=n_samples,
        tau=p["tau"], frac_zero=p["frac_zero"], sigma_zero=p["sigma_zero"],
        Av_max=p["Av_max"], Rv=p["Rv"],
        random_state=random_state,
    )


def sample_cosmological_redshift(n_samples=1, z_min=0.01, z_max=0.5, 
                                H0=70, Om=0.3, OL=0.7, random_state=None):
    """
    Genera muestras de redshift usando distribución volumétrica cosmológica.
    
    Parámetros:
    -----------
    n_samples : int
        Número de muestras a generar
    z_min, z_max : float
        Rango de redshift
    H0 : float
        Constante de Hubble (km/s/Mpc)
    Om, OL : float
        Parámetros cosmológicos Ω_m y Ω_Λ
    random_state : int, opcional
        Semilla para reproducibilidad
        
    Retorna:
    --------
    z_samples : array
        Valores de redshift muestreados
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Elemento de volumen comóvil correcto: dV/dz ∝ D_C(z)² / E(z), con
    # D_C(z) = (c/H0)∫₀ᶻ dz'/E(z') la distancia comóvil. La constante c/H0 se
    # cancela en la CDF normalizada, así que no hace falta.
    # FIX 2026-06-28: el código previo usaba (1+z)² en vez de D_C(z)², lo que
    # aplanaba la distribución (~uniforme) y sesgaba el muestreo hacia z bajo
    # (~8× exceso de SNe cercanas). Verificado contra astropy
    # FlatLambdaCDM.differential_comoving_volume. Ver bitácora Proyección 2026-06-28.
    z_grid = np.linspace(0.0, z_max, 2000)          # desde 0 para integrar D_C
    E = np.sqrt(Om * (1 + z_grid)**3 + OL)
    inv_E = 1.0 / E
    # D_C(z) por integración acumulada (trapecio) de 1/E desde 0
    D_C = np.concatenate([[0.0], np.cumsum(0.5 * (inv_E[1:] + inv_E[:-1]) * np.diff(z_grid))])
    dV_dz = D_C**2 / E

    # Restringir el muestreo al rango pedido [z_min, z_max] (volume-weighted dentro del rango)
    mask = z_grid >= z_min
    zg, w = z_grid[mask], dV_dz[mask]

    # CDF normalizada + muestreo por inversión
    cdf = np.cumsum(w)
    cdf = cdf / cdf[-1]
    u_samples = np.random.random(n_samples)
    z_samples = np.interp(u_samples, cdf, zg)

    return z_samples


def redden_spectrum_adjusted(la, spec, Rv, ebmv, norm='n'):
    """
    Version en python de codigo en IDL de G. Pignata 2004

    Aplica corrección de enrojecimiento o desenrojecimiento a un espectro dado,
    basándose en la ley de Cardelli. La función también ofrece la opción de normalizar
    el espectro corregido.

    Parámetros:
    la (array de numpy): Array de longitudes de onda en Ångströms.
    spec (array de numpy): Espectro original que se desea corregir.
    Rv (float): Relación de extinción total a selectiva (R_V).
    ebmv (float): Exceso de color E(B-V). Valores positivos para enrojecimiento,
                  negativos para desenrojecimiento.
    norm (str): 'y' para normalizar el espectro corregido, cualquier otro valor
                para no normalizar.

    Devuelve:
    array de numpy: Espectro corregido.

    Notas:
    - La corrección se aplica solo para longitudes de onda entre 3030 Å y 33000 Å.
    - Utiliza la ley de Cardelli para calcular la extinción.
    """

    # Verifica si hay longitudes de onda menores a 3030 Å, que no son válidas
    if np.min(la) <= 1250:
        raise ValueError("No hay corrección para longitudes de onda menores de 1250")

    # Inicializa abs_flux como un arreglo de unos, del mismo tamaño que spec
    abs_flux = np.ones_like(spec)

    # Corrección en el infrarrojo (IR)
    la_ir_indices = np.where((la < 33000) & (la > 9000))[0]
    if len(la_ir_indices) > 0:
        # Calcula y aplica la corrección IR a las partes relevantes de abs_flux
        la_ir = la[la_ir_indices]
        x_ir = 1.0 / (la_ir / 10000.0)
        a_ir = 0.574 * x_ir ** 1.61
        b_ir = -0.527 * x_ir ** 1.61
        abs_flux[la_ir_indices] = 10 ** (-0.4 * ((a_ir * Rv + b_ir) * ebmv))

    # Corrección óptica
    la_opt_indices = np.where((la <= 9000) & (la > 3000))[0]
    if len(la_opt_indices) > 0:
        # Calcula y aplica la corrección óptica
        la_opt = la[la_opt_indices]
        x_opt = 1.0 / (la_opt / 10000.0)
        y = x_opt - 1.82
        a_opt = 1.0 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
        b_opt = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
        abs_flux[la_opt_indices] = 10 ** (-0.4 * ((a_opt * Rv + b_opt) * ebmv))
        
    # Corrección UV
    la_uv_indices = np.where((la <= 3000) & (la >= 1250))[0]
    if len(la_uv_indices) > 0:
        la_uv = la[la_uv_indices]
        x_uv = 1.0 / (la_uv / 10000.0)  # Convertir a micrómetros y luego tomar el inverso

        # Calcula Fa(x) y Fb(x)
        Fa = np.where((x_uv > 5.9) & (x_uv < 8), -0.04473 * (x_uv - 5.9)**2 - 0.009779 * (x_uv - 5.9)**3, 0)
        Fb = np.where((x_uv > 5.9) & (x_uv < 8), 0.2130 * (x_uv - 5.9)**2 + 0.1207 * (x_uv - 5.9)**3, 0)

        # Calcula a(x) y b(x)
        a_uv = 1.752 - 0.316 * x_uv - 0.104 / ((x_uv - 4.67)**2 + 0.341) + Fa
        b_uv = -3.090 + 1.825 * x_uv + 1.206 / ((x_uv - 4.62)**2 + 0.263) + Fb

        # Aplica la corrección UV
        abs_flux[la_uv_indices] = 10 ** (-0.4 * ((a_uv * Rv + b_uv) * ebmv))
        

    # Normaliza abs_flux si se requiere
    if norm == 'y':
        abs_flux /= np.mean(abs_flux) if np.mean(abs_flux) > 0 else -np.mean(abs_flux)

    # Calcula el espectro de salida multiplicando el espectro original por abs_flux
    spec_out = spec * abs_flux
    return spec_out

def correct_redeening(sn,ESPECTRO,fases,ebmv=None,ebmv_host=None,ebmv_mw=None,mu=None,z=None,path_save='',reverse=False,to_abs_mag=False,write=False,use_DL=False,ubuntu=False):
    '''
    El codigo busca el archivo modulus donde esta toda la info de las SN
    sn_list: lista de las SN con la extension para leerlas, ex ['SN2005cs.dat','SN2013ej.dat']
    names: Lista con los nombres de la SN sin la extencion, para buscar el nombre en archivo modulus
    path_save: path donde se guarda el nuevo espectro, solos si write ==True
    reverse: Si es False, corrigue por reddening, si es True, Agrega el reddening (proceso inverso)
    to_abs_mag= Si es True, lleva todo a magnitud absoluta o corrigue por ella

    '''

    from config import MODULUS_PATH
    modulus_path=str(MODULUS_PATH)
    modulus=pd.read_csv(modulus_path)

    print(sn)
    name=sn
    NEW_ESPECTRO=[]
    if mu ==None:
        mu=float(modulus[modulus['Sn']==name]['modulus'])
    else:
        mu=mu
    
    # Manejar extinción: usar separados o total
    if ebmv_host is not None and ebmv_mw is not None:
        # Usar valores separados (físicamente correcto)
        ebmv_host_val = ebmv_host
        ebmv_mw_val = ebmv_mw
        ebmv = ebmv_host + ebmv_mw  # Solo para compatibilidad/log
    elif ebmv is not None:
        # Usar valor total (método anterior)
        ebmv_host_val = ebmv  # Asumir que es extinción total
        ebmv_mw_val = 0.0
    else:
        # Obtener del archivo de datos
        ebmv = float(modulus[modulus['Sn']==name]['ebmv'])
        ebmv_host_val = 0
        ebmv_mw_val = ebmv
    if z!=None:
        z=z
    else:
        z=float(modulus[modulus['Sn']==name]['z'])
    print("E(B-v),Z,mu")
    print(ebmv,z,mu)
    
    for i in range(len(ESPECTRO)):
        
        df=pd.DataFrame({'wave':ESPECTRO[i].wave,'flux':ESPECTRO[i].flux})
        
        df = df.reset_index(drop=True)
        #print(i,df)
        
        
        if reverse == False:
            # 1. Quitar extinción de la Vía Láctea (aplicada en marco observado)
            if ebmv_mw_val > 0:
                df['flux'] = redden_spectrum_adjusted(df['wave'], df['flux'], Rv=3.1, ebmv=-ebmv_mw_val)

            # 2. Quitar redshift → pasar a rest-frame
            df['wave'] = df['wave'] / (1 + z)
            df['flux'] = df['flux'] * (1 + z) 
            # 3. Quitar extinción del host (en rest-frame)
            if ebmv_host_val > 0:
                df['flux'] = redden_spectrum_adjusted(df['wave'], df['flux'], Rv=3.1, ebmv=-ebmv_host_val)

            # 4. Llevar a 10 pc si to_abs_mag = True
            if to_abs_mag:
                d_pc = 10**((mu + 5) / 5)
                df['flux'] = df['flux'] * ((d_pc / 10.0) ** 2)

            # 5. Regrillar
            xnew = np.arange(int(df.wave.min()), int(df.wave.max()) + 1, 1)
            interpolated_flux = interpolate.interp1d(df.wave, df.flux, fill_value="extrapolate")
            new_df = pd.DataFrame({'wave': xnew, 'flux': interpolated_flux(xnew)})




        if reverse==True: #aplicamos todos los procesos de manera inversa.
            # ORDEN FÍSICO CORRECTO para añadir efectos observacionales:
            # SN intrinsic → Host extinction → Redshift → MW extinction → Distance
            
            # 1. Aplicar extinción del host (más cercana a la SN)
            if ebmv_host_val > 0:
                new_spectro=redden_spectrum_adjusted(df.wave,df.flux,Rv=3.1,ebmv=ebmv_host_val)
                df['flux']=new_spectro
            
            # 2. Aplicar redshift cosmológico
            if z != 0:
                df['wave'] = df['wave'] * (1 + z)
                df['flux'] = df['flux'] / (1 + z)
            # 3. Aplicar extinción de la Vía Láctea (más externa)
            if ebmv_mw_val > 0:
                new_spectro=redden_spectrum_adjusted(df.wave,df.flux,Rv=3.1,ebmv=ebmv_mw_val)
                df['flux']=new_spectro
            
            # 4. Aplicar efectos de distancia al final
            if to_abs_mag==True and use_DL==False:
                d_pc=10**((mu+5)/5) #distancia en parsec
                df['flux']=df['flux']/((d_pc/10)**2) #correguimos la magnitud absoluta
            elif to_abs_mag==False and use_DL==True:
                DL=DL_calculator(z)
                df['flux']=df['flux'] * ((1e-5 /DL)**2) #el 1e-5 son 10pc, los pasamos a Mpc para que tenga las mimas unidades de DL, Fobs= Fem*(1e-5/DL)^2
            elif to_abs_mag==True and use_DL==True:
                print('No puedes usar to_abs_mag=True y use_DL True al mismo tiempo\nporfavor decide si correguir por mag abs (modulo de distancia) o llevar a un nuevo redshift y calcular DL')
                return None,None
            

            #regrillamos
            #print(int(min(df.wave)),max(df.wave)+1)
            xnew=np.arange(int(min(df.wave)),max(df.wave)+1,1)
            interpolated_flux = interpolate.interp1d(df.wave,df.flux,fill_value = "extrapolate")
            
            new_df = pd.DataFrame({'wave': xnew,'flux':interpolated_flux(xnew)})

        
        
        NEW_ESPECTRO.append(new_df)
    
    if write==True:
        write_spetra(NEW_ESPECTRO,fases,os.path.join(path_save,sn))  # Función no implementada
        pass

            

    
    return NEW_ESPECTRO,fases


def write_spetra(ESPECTRO,fases,path):
    #ahora escribimos el espectro promediado
    file1=open(path,'w')
    
    for j in range(len(fases)):

        file1.write('# time:\t'+str(fases[j])+'\n')
        file1.write('# SPEC \n')
        file1.write('#      WAVE   FLUX')
        df_spec=ESPECTRO[j]
        for ii in range(len(df_spec)):
            file1.write('\n')
            file1.write(str(df_spec.iloc[ii]['wave'])+'\t'+str(float(df_spec.iloc[ii]['flux'])))
        file1.write('\n')
    file1.close()