import numpy as np
import pandas as pd
import os
from scipy import interpolate
import matplotlib.pyplot as plt
from math import *
from math import sqrt, sin, sinh, exp

def DL_calculator(z, H0=69.6, WM=0.286, WV=0.714, n=1000):
    """
    Calcula la distancia de luminosidad (DL) en Mpc a partir de un redshift 'z' y 
    parámetros cosmológicos:
    - H0: Constante de Hubble (por defecto 69.6 km/s/Mpc)
    - WM: Densidad de materia (Omega_M)
    - WV: Densidad de energía oscura (Omega_Lambda)
    - n : Número de pasos para la integración (por defecto 1000)

    Usa la convención de la Cosmology Calculator de Ned Wright.
    """

    # Constantes físicas
    c = 299792.458  # Velocidad de la luz en km/s
    h = H0 / 100.0
    WR = 4.165e-5 / (h * h)  # Densidad de radiación (3 neutrinos masivos)
    WK = 1.0 - WM - WR - WV  # Curvatura
    az = 1.0 / (1.0 + z)

    # Cálculo de la distancia comóvil radial (DCMR)
    DCMR = 0.0
    for i in range(n):
        a = az + (1 - az) * (i + 0.5) / n
        adot = sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        DCMR += 1.0 / (a * adot)
    DCMR *= (1.0 - az) / n

    # Cálculo del ajuste de curvatura (transversal)
    x = sqrt(abs(WK)) * DCMR
    if x > 0.1:
        if WK > 0:
            ratio = sinh(x) / x
        else:
            ratio = sin(x) / x
    else:
        y = x * x
        if WK < 0:
            y = -y
        ratio = 1.0 + y / 6.0 + (y * y) / 120.0

    # Distancia comóvil transversal
    DCMT = ratio * DCMR

    # Distancia angular y de luminosidad
    DA = az * DCMT
    DL = DA / (az * az)

    # Conversión final a Mpc
    DL_Mpc = (c / H0) * DL

    return DL_Mpc




def leer_spec(path,ot=False,MJD=False,as_pandas=False,compress=False):

    if compress==True:
        import gzip

        with gzip.open(path, 'rb') as f2:
            espectro_lineas= [linea.split() for linea in f2]
    else:
        with open(path,'rt') as f2:
            espectro_lineas= [linea.split() for linea in f2]

    componente_filas_spec=[]
    spec=[]
    fases=[]
    iniciales=[]
    finales=[]
    alphas=[]
    for i in range(len(espectro_lineas)):
        if len(espectro_lineas[i])>1:
            if 'SPEC' in espectro_lineas[i][1] and 'NSPEC' not in espectro_lineas[i][1]:
                componente_filas_spec.append(i)  #me dice en que componente esta el spectro de la posicion

            if 'time:' == espectro_lineas[i][1] :
                fase=espectro_lineas[i][2]

                spec.append(float(fase)) #le saque el round porque me daba espectros repetidos cuando, si lo quiero poner deberia quiza usar un drop duplicates en el
                                            #calculo del rms y mad
                fases=spec
                #print(fase)
            else:
                if MJD==True:
                    if len(espectro_lineas[i])==5:
                        if 'MJD' in espectro_lineas[i][4] and 'SPEC' in espectro_lineas[i]:
                            fase=espectro_lineas[i][4]
                            fase=float(fase.strip('MJD='))
                            spec.append(fase) #saque el round
                            fases=spec
            if ot==True:
                if 'ini:' in espectro_lineas[i][1]:
                    inicial=espectro_lineas[i][2]
                    iniciales.append(float('%.3f'%float(inicial)))
                if 'fin:' in espectro_lineas[i][1]:
                    final=espectro_lineas[i][2]
                    finales.append(float('%.3f'%float(final)))
                if 'alpha:' in espectro_lineas[i][1]:
                    alpha=espectro_lineas[i][2]
                    alphas.append(float('%.3f'%float(alpha)))


    ESPECTRO=[]
    for j in range(len(componente_filas_spec)):
        #nombre_spec = str(spec[j])
        #print (nombre_spec)
        espectro1=[]
        if j==len(componente_filas_spec)-1:
            final=len(espectro_lineas)
            lineas_espectro=np.arange(componente_filas_spec[j]+1,final,1)
            #print (lineas_espectro)
            for i in lineas_espectro:
                if espectro_lineas[i][1]=='WAVE':
                    continue
                if float(espectro_lineas[i][1])!=0:
                    espectro1.append(espectro_lineas[i])

        else:
            if ot==True:
                res=4 #esto es para que lea los archivos bien ya que los OT tienen info extra
            else:
                res=1
            lineas_espectro=np.arange(componente_filas_spec[j]+1,componente_filas_spec[j+1]-res,1) #me dice las lineas que tengo q guardar por filtro
            #print (lineas_espectro)
            for i in lineas_espectro:

                if len(espectro_lineas[i])==0:
                    break
                if espectro_lineas[i][1]=='WAVE':
                    continue

                if float(espectro_lineas[i][1])!=0:
                    espectro1.append(espectro_lineas[i])
        if as_pandas==True:
            #print(len(espectro1[0]))
            #print(espectro1)
            try:
                if len(espectro1[0])==2:
                    espectro1=pd.DataFrame(espectro1,dtype="float64",columns=["wave","flux"])
                elif len(espectro1[0])==3:
                    espectro1=pd.DataFrame(espectro1,dtype="float64",columns=["wave","flux",'fluxerr'])
                else:
                    return print('Imposible llevar a pandas, revisa las columnas')
            except:
                espectro1=pd.DataFrame(espectro1,dtype="float64",columns=["wave","flux"])

        ESPECTRO.append(espectro1)
    #print('Phases:',fases)

    if ot==True:
        return ESPECTRO,fases,iniciales,finales,alphas
    else:
        return ESPECTRO,fases






def Syntetic_photometry_v2(xspec,yspec,xband,yband):
    #recibe pandas dataframe o arrays, y calcula la inrterseccion
    #pueden ser de distinto largo y la respuesta de la banda se adapta a la grilla del espectro con regrid

    from sklearn.metrics import auc

    ######################################################################
    # AHORA VAMOS A CALCULAR EL AREA DE LA RESPUESTA PARA NORMALIZAR
    a_banda=auc(xband,yband) #esta es el area de la banda

    #necesitamos el area de la interseccion##############################
    import scipy
    from scipy.interpolate import griddata
    regrid=scipy.interpolate.griddata(xband, yband,xspec, method='linear', rescale=False)
    df_regrid=pd.DataFrame({'wave':xspec,'response':regrid})
    df_regrid = df_regrid.dropna(how='any') #eliminamos los nan

    try:
        a_inter=auc(df_regrid['wave'],df_regrid['response'])
    except ValueError:
        return 0,0
    #####################################################################
    porcentaje=a_inter/a_banda
    #print(a_inter,a_banda)
    #print('porcentaje:',a_inter/a_banda)
    spec1_df=pd.DataFrame({'wave':xspec,'flux':yspec})
    #ahora calculamos el numerador de la integral
    merge = pd.merge(spec1_df, df_regrid, how='inner', on=['wave']) #mismo q df_regrid pero con el flujo del spec

    num=auc(merge['wave'],merge['flux']*merge['response']*merge['wave'])

    #calculamos el denominador

    den=auc(xband,yband*xband) #este es el numerador para normalizar

    total_=num/den
    #print('area :',total_, 'porcentaje:',porcentaje)
    #print('num|den')
    #print(num,den)
    

    return total_,porcentaje



def maximo_lc(tipo,sn,ubuntu=False):
    if tipo=='II':
        if ubuntu==True:
            maximum_df=pd.read_csv("/mnt/g/Mi unidad/Work/OT_unidos_2/OT_combines_test/maximum_II.txt")

        else:
            maximum_df=pd.read_csv("G:\Mi unidad\Work\OT_unidos_2\OT_combines_test\maximum_II.txt")
        max_=float(maximum_df.loc[maximum_df['name']==sn]['fase'])
        return max_
    if tipo=='Ia':
        if ubuntu==True:
             
            df_max=pd.read_csv("/mnt/g/Mi unidad/Work/OT_unidos_2/OT_combines_test/maximum_Ia.txt")
        else:
            df_max=pd.read_csv("G:\Mi unidad\Work\OT_unidos_2\OT_combines_test\maximum_Ia.txt")
        df_max[df_max.name==sn]

    if tipo=='Ibc' or tipo=='Ib' or tipo=='Ic':
            df_max=pd.read_csv(r"G:\Mi unidad\Work\Universidad\Phd\Practica2\maximum_Ibc.dat")
            #df_max=pd.read_csv( "G:\Mi unidad\Spectral Time Series - Ramirez M\Data\maximum_perband.dat")
            df_max[df_max.name==sn]
    
    band_system=['V','R','r','g','i','I','B','U','u','z'] #We use this order to obtain the day of the maximum
    for elemento in band_system:

        maximum=df_max.loc[(df_max['name']==sn) & (df_max['filter']==elemento )]['mjd_max']
        if len(maximum)>0:
            maximum=float(maximum)
            break
    return maximum


def Loess_fit(FILTRO_,V,mag_to_flux=False,interactive=False,fig_title='',use_cte='True',alpha=0,corte=40,plot=True): #entra solo una curva
    cteB=6.460803024157998e-09
    cteV=3.67558126207669e-09
    cteR=2.2319872881446082e-09
    cteI=1.1769629663231628e-09
    cteU=4.346809582387867e-09

    cteu=9.512689427618875e-09#859.5e-11
    cteg=4.791168585588861e-09#466.9e-11
    cter=2.818556152283821e-09#278.0e-11
    ctei=1.9069984363571034e-09#185.2e-11
    ctez=1.413365541656362e-09#131.5e-11
    arr_ctes=['cteB','cteV','cteR','cteI','cteU','cteu','cteg','cter','ctei','ctez']
    arr_val_ctes=[cteB,cteV,cteR,cteI,cteU,cteu,cteg,cter,ctei,ctez]
    constante='cte'+V #pongo V pero es cualquier banda
    # Print eliminado para evitar duplicación
    for jj in range(len(arr_ctes)):
            if constante==arr_ctes[jj]: #recorremos el arreglo de constantes en busca de la que le corresponde
                # Print eliminado para evitar duplicación  
                mul=arr_val_ctes[jj]
                if use_cte=='False':
                    mul=1




    lc_df=pd.DataFrame( FILTRO_) #data frame de la curva real
    lc_df.rename(columns={0:'mjd',1:'flux',2:'flux_err',
                          3:'Upperlimit',4:'Instrument',5:'Telescope',6:'Source'},inplace=True)
    #lc_df['flux_err'].replace(0,0.05,inplace=True)

    #if len(lc_df.keys())>3:
     #   lc_df=lc_df.loc[lc_df.loc[:, 'Upperlimit'] !='T'] #si ES T es upperlimit, por lo que se elimina del df

    DJ=lc_df['mjd']
    if len(DJ)<5:
        print('No hay suficientes datos (5) datos en la banda para realizar el fit:',V)
        ALR_df=[]
        return ALR_df

    #return lc_df
    if mag_to_flux==True:
        lc_df['flux']=10**(-2*lc_df['flux']/5)
        lc_df['flux_err']=(lc_df['flux_err']*lc_df['flux'])/1.086
        lc_df['flux_err'].replace(0,0.05,inplace=True)
    lc_df['flux_err'].replace(0,np.max(lc_df['flux'])*0.1,inplace=True)
    flux=lc_df['flux']*mul
    flux_err=lc_df['flux_err']*mul

    import sys
    sys.path.append('/home/mauri/Escritorio/ALR/src')
    sys.path.append('/data1/mauricio/ALR/src')
    sys.path.append("G:\\Mi unidad\\Work\\ALR\src")
    from ALR import Automated_Loess_Regression
    import plot_ALR

    save_interactive='n'
    
    while save_interactive=='n':
        if len(alpha)==1:

            ALR = Automated_Loess_Regression(DJ, flux,err_y=flux_err,alpha=alpha[0])
            if min(DJ)<0 and max(DJ)>0:x_interp=np.arange(int(min(DJ)),int(max(DJ)+1)) # -> el +2 es para que calze con "sampleo"
            if min(DJ)>0 and max(DJ)>0:x_interp=np.arange(int(min(DJ)),int(max(DJ)+1))
            if min(DJ)<0 and max(DJ)<0:x_interp=np.arange(int(min(DJ)+1),int(max(DJ)))
            if min(DJ)>0 and max(DJ)<0: print('Checkea los dias julianos')
            #print(DJ)
            #print(x_interp)
            if len(x_interp)<5:
                return print('No se recomienda hacer Loess con menos de 5')
            y_ALR,y_err_ALR=ALR.interp(x_interp) #este es para el plot

            #print(x_df['mjd'])
            #print('el loess',y_ALR2)

            x_interp_df=pd.DataFrame(x_interp,columns=['mjd'])
            y_ALR_df=pd.DataFrame(y_ALR,columns=['flux'])
            y_err_ALR_df=pd.DataFrame(y_err_ALR,columns=['flux_err'])
            x_interp_df.reset_index(drop=True,inplace=True)
            ALR_df=pd.merge(x_interp_df, y_ALR_df, right_index=True, left_index=True)
            ALR_df=pd.merge(ALR_df, y_err_ALR_df, right_index=True, left_index=True)


            y_ALR2,y_err_ALR2=ALR.interp(DJ) #para los residuales
        
        if len(alpha)==2:
            print('ENTRANDO AL DIVIDIDO')

            #--------------
            lc_df1=lc_df[:corte]
            lc_df2=lc_df[corte:]
            #------------
            DJ1=lc_df1['mjd']
            flux1=lc_df1['flux']
            flux_err1=lc_df1['flux_err']
            lc_upper_df1=lc_df1.loc[lc_df1.loc[:, 'Upperlimit'] =='T']
            flux_upper1=lc_upper_df1['flux']
            flux_err_upper1=lc_upper_df1['flux_err']
            if len(DJ1)<5:
                print('No hay suficientes datos (5) datos en la banda para realizar el fit:',V)
                ALR_df=[]
                return ALR_df
                continue
            
            DJ2=lc_df2['mjd']
            flux2=lc_df2['flux']
            flux_err2=lc_df2['flux_err']
            lc_upper_df2=lc_df2.loc[lc_df2.loc[:, 'Upperlimit'] =='T']
            flux_upper2=lc_upper_df2['flux']
            flux_err_upper2=lc_upper_df2['flux_err']
            if len(DJ2)<5:
                print('No hay suficientes datos (5) datos en la banda para realizar el fit:',V)
                ALR_df=[]
                return ALR_df
                continue
            
            #-----------
            ALR1=Automated_Loess_Regression(DJ1, flux1,err_y=flux_err1,alpha=alpha[0])
            ALR2=Automated_Loess_Regression(DJ2, flux2,err_y=flux_err2,alpha=alpha[1])
            
            if min(DJ1)<0 and max(DJ1)>0:x_interp1=np.arange(int(min(DJ1)),int(max(DJ1)+1)) # -> el +2 es para que calze con "sampleo"
            elif min(DJ1)>0 and max(DJ1)>0:x_interp1=np.arange(int(min(DJ1)),int(max(DJ1)+1))
            elif min(DJ1)<0 and max(DJ1)<=0:x_interp1=np.arange(int(min(DJ1)),int(max(DJ1)+1)) # Incluye el caso max=0
            else: 
                print(f'Error: Condición no contemplada en los días julianos DJ1: min={min(DJ1)}, max={max(DJ1)}')
                ALR_df=[]
                return ALR_df
                
            if min(DJ2)<0 and max(DJ2)>0:x_interp2=np.arange(int(min(DJ2)),int(max(DJ2)+1)) # -> el +2 es para que calze con "sampleo"
            elif min(DJ2)>0 and max(DJ2)>0:x_interp2=np.arange(int(min(DJ2)),int(max(DJ2)+1))
            elif min(DJ2)<0 and max(DJ2)<=0:x_interp2=np.arange(int(min(DJ2)),int(max(DJ2)+1)) # Incluye el caso max=0
            else: 
                print(f'Error: Condición no contemplada en los días julianos DJ2: min={min(DJ2)}, max={max(DJ2)}')
                ALR_df=[]
                return ALR_df
            #print('DJ:',DJ)
            #print('flux;',flux)
            #print('x para interpolar:',x_interp)
            if len(x_interp1)<5:
                print('No se recomienda hacer Loess con menos de 5')
                continue
            if len(x_interp2)<5:
                print('No se recomienda hacer Loess con menos de 5')
                continue
            
            y_ALR1,y_err_ALR1=ALR1.interp(x_interp1) #este es para el plot
            y_ALR2,y_err_ALR2=ALR2.interp(x_interp2) #este es para el plot
            y_ALR=np.concatenate((y_ALR1, y_ALR2), axis=None)
            y_err_ALR=np.concatenate((y_err_ALR1, y_err_ALR2), axis=None)


            #print(x_df['mjd'])
            #print('el loess',y_ALR2)
            
            x_interp_df1=pd.DataFrame(x_interp1,columns=['mjd'])
            x_interp_df2=pd.DataFrame(x_interp2,columns=['mjd'])
            x_interp_df=pd.concat([x_interp_df1,x_interp_df2])
            
            y_ALR_df1=pd.DataFrame(y_ALR1,columns=['flux'])
            y_ALR_df2=pd.DataFrame(y_ALR2,columns=['flux'])
            y_ALR_df=pd.concat([y_ALR_df1,y_ALR_df2])

            y_err_ALR_df1=pd.DataFrame(y_err_ALR1,columns=['flux_err'])
            y_err_ALR_df2=pd.DataFrame(y_err_ALR2,columns=['flux_err'])
            y_err_ALR_df=pd.concat([y_err_ALR_df1,y_err_ALR_df2])

            
            x_interp_df.reset_index(drop=True,inplace=True)
            y_ALR_df.reset_index(drop=True,inplace=True)
            y_err_ALR_df.reset_index(drop=True,inplace=True)
            #print(y_ALR_df)
            #print(x_interp_df)
            #print(y_err_ALR_df)
            ALR_df=pd.merge(x_interp_df, y_ALR_df, right_index=True, left_index=True)
            #print(ALR_df)
            ALR_df=pd.merge(ALR_df, y_err_ALR_df, right_index=True, left_index=True)
            
            ALR_df.drop_duplicates(subset=['mjd'],inplace=True)
            #print(ALR_df)

            y_ALR_residuals1,y_err_residuals1=ALR1.interp(DJ1) #para los residuales
            y_ALR_residuals2,y_err_residuals2=ALR2.interp(DJ2[:]) #para los residuales
            
            y_ALR_residuals=np.concatenate((y_ALR_residuals1, y_ALR_residuals2), axis=None)

        #------------------------------------------------------------------
        if plot==True:
            fig = plt.figure(1908546223,figsize=(6,8))
            fig.subplots_adjust(left=0.12, bottom=0.05, hspace=0.1, right=0.99, top=0.96)
            ax_ = fig.add_subplot(211)

            ax_.errorbar(DJ,flux,yerr=flux_err,fmt='o',label='OBS',zorder=1)
            ax_.plot(x_interp_df['mjd'],y_ALR,'-',label='Loess fit',zorder=2)
            ax_.plot(x_interp_df['mjd'], y_ALR+3.0*y_err_ALR, '--', color='green'  , lw=1, zorder=1)
            ax_.plot(x_interp_df['mjd'], y_ALR-3.0*y_err_ALR, '--', color='green'  , lw=1, zorder=1)
            if mag_to_flux==False:
                plt.gca().invert_yaxis()
            plt.title('Filtro '+V)
            plt.xlabel(r'$\mathtt{Día}$')
            plt.ylabel(r'$\mathtt{Flux}$')
            if fig_title!='': plt.title('Filtro '+V+' '+fig_title)

            ax_.tick_params('both', length=6, width=1, which='major', direction='in')
            ax_.tick_params('both', length=3, width=1, which='minor', direction='in')
            ax_.xaxis.set_ticks_position('both')
            ax_.yaxis.set_ticks_position('both')
            plt.legend()
            #plt.xlim(min(DJ_band)-3,max(DJ_band)+3)
            #plt.rcParams['figure.figsize'] = [10, 8]
            #plt.figure(999)
            #--------------------------------------------------------------------------


            #fig = plt.figure(19085463,figsize=(9,9))
            #fig.subplots_adjust(left=0.12, bottom=0.05, hspace=0.0, right=0.99, top=0.96)
            ax  = fig.add_subplot(212)

            if len(alpha)==1:
                residuales=((flux-y_ALR2)/flux) #real-loess
            if len(alpha)==2:
                residuales=((flux-y_ALR_residuals)/flux) #real-loess

                

            ax.plot(DJ,np.zeros(len(DJ)),'--',ms=1,color='black')
            ax.plot(DJ,residuales,'o',ms=10,color='red')
            ax.tick_params('both', length=6, width=1, which='major', direction='in')
            ax.tick_params('both', length=3, width=1, which='minor', direction='in')
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.set_ylabel(r'$\mathtt{Residuals = \frac{F_{syn}-F_{ALR}}{F_{ALR}}}$')
            ax.set_xlabel(r'$\mathtt{MJD}$')
            #plt.xlim(min(DJ_band)-3,max(DJ_band)+3)
            #plt.legend()


            plt.show()

        if len(alpha)==1:
            print ('The alpha value for the ALR is:',ALR.alpha)
        if len(alpha)==2:
            print ('The alpha values for the ALR is:',ALR1.alpha,'and',ALR2.alpha)


        if interactive==True:
            save_interactive=input('Save? y/n')
            if save_interactive=='n':
                alpha=float(input('Input the alpha value: '))
            if save_interactive=='y':
                continue
            while save_interactive!='y' and save_interactive!='n':
                print('Pls use y/n notation.')
                save_interactive=input('Save? y/n')
        if interactive==False:
            save_interactive='y'


    return ALR_df


cteB=6.460803024157998e-09
cteV=3.67558126207669e-09
cteR=2.2319872881446082e-09
cteI=1.1769629663231628e-09
cteU=4.346809582387867e-09

cteu=9.512689427618875e-09#859.5e-11
cteg=4.791168585588861e-09#466.9e-11
cter=2.818556152283821e-09#278.0e-11
ctei=1.9069984363571034e-09#185.2e-11
ctez=1.413365541656362e-09#131.5e-11



