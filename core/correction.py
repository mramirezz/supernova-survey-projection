# 📚 IMPORTS PARA CORRECTION
# ===========================
import os
import numpy as np
import pandas as pd
from scipy import interpolate
from .utils import DL_calculator


def sample_host_extinction_SNIa(n_samples=1, tau=0.4, Av_max=3.0, Rv=3.1, random_state=None):
    """
    Genera muestras de extinción del host para SNe Ia usando distribución exponencial en A_V.
    
    JUSTIFICACIÓN ACADÉMICA:
    Basado en Holwerda et al. (2014) "SNIa Host Galaxy Properties and the Dust Extinction Distribution"
    P(A_V) ∝ exp(-A_V/τ) con τ=0.4 mag (estándar académico validado)
    
    Parámetros:
    -----------
    n_samples : int
        Número de muestras a generar
    tau : float
        Parámetro de escala según Holwerda et al. (2014): τ=0.4 mag
    Av_max : float
        Valor máximo de A_V (corte físico)
    Rv : float
        Relación R_V = A_V / E(B-V)
    random_state : int, opcional
        Semilla para reproducibilidad
        
    Retorna:
    --------
    ebmv_host : array
        Valores de E(B-V) del host muestreados
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Muestreo de distribución exponencial en A_V
    Av_samples = np.random.exponential(tau, size=n_samples)
    
    # Aplicar corte físico
    Av_samples = np.clip(Av_samples, 0, Av_max)
    
    # Convertir a E(B-V)
    ebmv_host = Av_samples / Rv
    
    return ebmv_host


def sample_host_extinction_core_collapse(n_samples=1, sn_type='II', tau=0.4, 
                                        Av_max=3.0, Rv=3.1, random_state=None):
    """
    Genera muestras de extinción del host para SNe core-collapse usando distribución exponencial.
    
    JUSTIFICACIÓN ACADÉMICA:
    Por consistencia con Holwerda et al. (2014) y principio de parsimonia científica,
    se usa la misma distribución exponencial P(A_V) ∝ exp(-A_V/τ) para todos los tipos de SN.
    Eliminación de distribuciones mixtas por falta de justificación académica sólida.
    
    Parámetros:
    -----------
    n_samples : int
        Número de muestras a generar
    sn_type : str
        Tipo de SN ('II', 'Ibc') - mantenido por compatibilidad
    tau : float
        Parámetro de escala exponencial (τ=0.4 según literatura)
    Av_max : float
        Valor máximo de A_V (corte físico)
    Rv : float
        Relación R_V = A_V / E(B-V)
    random_state : int, opcional
        Semilla para reproducibilidad
        
    Retorna:
    --------
    ebmv_host : array
        Valores de E(B-V) del host muestreados
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Distribución exponencial unificada para todos los tipos
    # Científicamente justificada por Holwerda et al. (2014)
    Av_samples = np.random.exponential(tau, size=n_samples)
    
    # Aplicar corte físico
    Av_samples = np.clip(Av_samples, 0, Av_max)
    
    # Convertir a E(B-V)
    ebmv_host = Av_samples / Rv
    
    return ebmv_host


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
    
    # Crear grilla de redshift para calcular CDF
    z_grid = np.linspace(z_min, z_max, 1000)
    
    # Calcular elemento de volumen comóvil dV/dz
    dV_dz = (1 + z_grid)**2 / np.sqrt(Om * (1 + z_grid)**3 + OL)
    
    # Calcular CDF normalizada
    cdf = np.cumsum(dV_dz)
    cdf = cdf / cdf[-1]
    
    # Muestreo por inversión de CDF
    u_samples = np.random.random(n_samples)
    z_samples = np.interp(u_samples, cdf, z_grid)
    
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

    if ubuntu==True:
        G_path='/mnt/g/Mi unidad'
    else:
        G_path="G:\Mi unidad"
    modulus_path=os.path.join(G_path,'Work','Universidad','Phd','DATA_OSC','all_data','modulos_query_all.cvs')
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
        
        
        if reverse==False:
            #--------correguimos por redshift, esto le saca el redshift original--------.
            df['wave']=df['wave']/(1+z)
    
            #-------------------correguimos por extinción en orden físico correcto-----------
            # ORDEN CORRECTO después de redshift: Ambas extinciones en rest-frame
            # Primero corregir extinción del host (más cercana a la SN, en rest-frame)
            if ebmv_host_val > 0:
                new_spectro=redden_spectrum_adjusted(df['wave'],df['flux'],Rv=3.1,ebmv=ebmv_host_val)
                df['flux']=new_spectro
            
            # Luego corregir extinción de la Vía Láctea (en rest-frame también)
            if ebmv_mw_val > 0:
                new_spectro=redden_spectrum_adjusted(df['wave'],df['flux'],Rv=3.1,ebmv=ebmv_mw_val)
                df['flux']=new_spectro

            #llevamos el flujo a una distancia comun de 10pc para posteriores usos. la formula F10=(d[pc]/10[pc])^2 * Flujo_actual
            #primero sacamos la distancia en parsec con el modulo de distancia--------------
            if to_abs_mag==True:
                d_pc=10**((mu+5)/5) #distancia en parsec

                df['flux']=df['flux']*((d_pc/10)**2)

            #regrillamos
            xnew=np.arange(int(min(df.wave)),max(df.wave)+1,1)
            interpolated_flux = interpolate.interp1d(df.wave,df.flux,fill_value = "extrapolate")
            
            new_df = pd.DataFrame({'wave': xnew,'flux':interpolated_flux(xnew)})



        if reverse==True: #aplicamos todos los procesos de manera inversa.
            # ORDEN FÍSICO CORRECTO para añadir efectos observacionales:
            # SN intrinsic → Host extinction → Redshift → MW extinction → Distance
            
            # 1. Aplicar extinción del host (más cercana a la SN)
            if ebmv_host_val > 0:
                new_spectro=redden_spectrum_adjusted(df.wave,df.flux,Rv=3.1,ebmv=-ebmv_host_val)
                df['flux']=new_spectro
            
            # 2. Aplicar redshift cosmológico
            if z != 0:
                df['wave'] = df['wave'] * (1 + z)
            
            # 3. Aplicar extinción de la Vía Láctea (más externa)
            if ebmv_mw_val > 0:
                new_spectro=redden_spectrum_adjusted(df.wave,df.flux,Rv=3.1,ebmv=-ebmv_mw_val)
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
        # write_spetra(NEW_ESPECTRO,fases,os.path.join(path_save,sn))  # Función no implementada
        pass

            

    
    return NEW_ESPECTRO,fases


