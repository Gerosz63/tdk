import datetime as dt
import numpy as np
import pandas as pd
import vppopt as vo
from matplotlib import pyplot as plt
import participants_errors as pe
import charts as c

globaldisplay = True
globalsummerize = False
globalcurrcat = 0


runtimelst = {"LP":list(), "MILP":list()}
MAElstp = {"LP":[list(), list(), list(), list()], "MILP":[list(), list(), list(), list()]}
RMSElstp = {"LP":[list(), list(), list(), list()], "MILP":[list(), list(), list(), list()]}
MAEcostlstp = {"LP":[list(), list(), list(), list()], "MILP":[list(), list(), list(), list()]}
MAElstn = {"LP":[list(), list(), list(), list()], "MILP":[list(), list(), list(), list()]}
RMSElstn = {"LP":[list(), list(), list(), list()], "MILP":[list(), list(), list(), list()]}
MAEcostlstn = {"LP":[list(), list(), list(), list()], "MILP":[list(), list(), list(), list()]}
nanlst = {"LP":list(), "MILP":list()}
costlstl = {"LP":list(), "MILP":list()}

def display_day(date:dt.datetime, data=None):
     if data is None:
          data = pd.read_csv("./data/cleard_data.csv", sep=',', dtype={"Gáz (fosszilis) erőművek": 'Float32', "Szárazföldi szélerőművek": 'Float32', "Naperőművek": 'Float32'}, parse_dates=[0])   
          data = data.loc[((data["Időpont"].dt.date == date.date()) & (data["Időpont"].dt.time != pd.to_datetime("00:00:00").time())) | ((data["Időpont"].dt.date == date.date() + dt.timedelta(days=1) ) & (data["Időpont"].dt.time == pd.to_datetime("00:00:00").time()))]
     sum = data["Szumma"].to_numpy()
     ge = data["Gáz (fosszilis) erőművek"].to_numpy()
     wp = data["Szárazföldi szélerőművek"].to_numpy()
     sp = data["Naperőművek"].to_numpy()

     sum = np.insert(sum, 0, sum[0])
     ge = np.insert(ge, 0, ge[0])
     sp = np.insert(sp, 0, sp[0])
     wp = np.insert(wp, 0, wp[0])
     plt.figure(figsize=(20,15))
     plt.step(range(25), sum, label="Szumma", color="red")
     plt.step(range(25), ge, label="Gáz (fosszilis) erőművek", color="blue")
     plt.step(range(25), wp, label="Szárazföldi szélerőművek", color="orange")
     plt.step(range(25), sp, label="Naperőművek", color="green")
     times = data['Időpont'].dt.strftime("%H:%M:%S").to_numpy()
     times = np.insert(times, 0, "00:00:00")
     times[-1] = "24:00:00"

     plt.xticks(range(25), times.tolist(),rotation=315)
     plt.legend()
     plt.title(data['Időpont'].dt.date.iloc[0].strftime("%Y.%m.%d Napi adat"))

def test(lp_items:list, milp_items:list, l:np.ndarray|None=None, date:dt.datetime|None=None, T:int=24, detaild_lp_items:list=[], detaild_milp_items:list=[], to_file:bool|list[bool]=False, lp_checker:np.ndarray|None=None, milp_checker:np.ndarray|None=None, disp_day_chart:bool=True):
     if l is None and date is None:
          raise Exception("Az l és a date értéke nem lehet egyszerre None!")
     if date is not None:
          data = pd.read_csv("./data/cleard_data.csv", sep=',', dtype={"Gáz (fosszilis) erőművek": 'Float32', "Szárazföldi szélerőművek": 'Float32', "Naperőművek": 'Float32'}, parse_dates=[0])   
          data = data.loc[((data["Időpont"].dt.date == date.date()) & (data["Időpont"].dt.time != pd.to_datetime("00:00:00").time())) | ((data["Időpont"].dt.date == date.date() + dt.timedelta(days=1) ) & (data["Időpont"].dt.time == pd.to_datetime("00:00:00").time()))]
          l = data["Szumma"].to_numpy()
          
          if disp_day_chart:
               display_day(date, data)
     if to_file == True or to_file == False:
          arr = [to_file, to_file]
          to_file = arr
     res_lp = vo.solve(lp_items, T, l, np.zeros_like(l), (lp_checker is not None) or to_file[0] == True, detaild_lp_items is not [], detaild_lp_items, True, "./out_lp.txt" if to_file[0] == True else None, lp_checker, "LP solver", False)
     res_milp = vo.solve(milp_items, T, l, np.zeros_like(l), (milp_checker is not None) or to_file[1] == True, detaild_milp_items is not [], detaild_milp_items, True, "./out_milp.txt" if to_file[1] == True else None, milp_checker, "MILP solver", False)
     
     print("LP solver eredménye:\t", end="")
     if res_lp[0]:
          print("Sikeres\nMegoldás: ", res_lp[1].x)
     else:
          print("Sikertelen")
     
     print("MILP solver eredménye:\t", end="")
     if res_milp[0]:
          print("Sikeres\nMegoldás: ", res_milp[1].x)
     else:
          print("Sikertelen")

# Függvény teszteléshez:
# A tesztelés során két futtatás csinál a program melyek a paraméterül kapott 2 részvevő halmazt veszik alapul.
# A futtatás során a fogyasztási görbe a T értéke megegyezik.
# Ezek után összehasonlításra kerülnek az eredmények
# Return (Sikeresség, (MAE, MSE, 1. költsége, 2. költsége))
def test_diff(i1:list[vo.VPPItem], i2:list[vo.VPPItem], T:int=24, l:np.ndarray=vo.gen_random_demand(24, 0, 20000), d:np.ndarray=np.zeros(24), display=False, disp_part:dict[list[int]]|None=None, display_compare=True):
     # A két futtatás
     res1 = vo.solve(i1, T, l, d, display=display, return_res=True, detaild_chart=True)
     res2 = vo.solve(i2, T, l, d, display=display, return_res=True, detaild_chart=True)
     
     # Összehasonlítás, ha mindekttő sikeres
     if res1[0] and res2[0]:

          # Számlálok látrehozása
          # Ezek mátrixok, melyek esetén az adott típusú 
          gasengines_x1 = np.zeros(0)
          gasengines_x2 = np.zeros(0)

          storages_x1 = np.zeros(0)
          storages_x2 = np.zeros(0)

          solars_x1 = np.zeros(0)
          solars_x2 = np.zeros(0)

          gasengines_x1_cost = np.zeros(0)
          gasengines_x2_cost = np.zeros(0)

          storages_x1_cost = np.zeros(0)
          storages_x2_cost = np.zeros(0)

          solars_x1_cost = np.zeros(0)
          solars_x2_cost = np.zeros(0)

          # Az első optimalizáció eredményeinek feldolgozása
          counter = 0
          for e in i1:
               if isinstance(e, vo.VPPEnergyStorage):
                    storages_x1_cost = np.vstack((storages_x1_cost, np.sum((res1[1].x[counter:counter + 3 * T] * res1[2][counter:counter + 3 * T]).reshape(3,-1), axis=0))) if storages_x1_cost.size != 0 else np.array(np.sum((res1[1].x[counter:counter + 3 * T] * res1[2][counter:counter + 3 * T]).reshape(3,-1), axis=0), ndmin=2)
                    storages_x1 = np.vstack((storages_x1,res1[1].x[counter + 2 * T:counter + 3 * T])) if storages_x1.size != 0 else np.array(res1[1].x[counter + 2 * T:counter + 3 * T], ndmin = 2)
                    counter += 3 * T
               elif isinstance(e, vo.VPPGasEngine):
                    if res1[3]:
                         gasengines_x1_cost = np.vstack((gasengines_x1_cost, np.sum((res1[1].x[counter:counter + 4*T] * res1[2][counter:counter + 4*T]).reshape((T,-1)), axis=1))) if gasengines_x1_cost.size != 0 else np.array(np.sum((res1[1].x[counter:counter + 4*T] * res1[2][counter:counter + 4*T]).reshape((T,-1)), axis=1), ndmin=2)
                    else:
                         gasengines_x1_cost = np.vstack((gasengines_x1_cost, res1[1].x[counter:counter + T] * res1[2][counter:counter + T])) if gasengines_x1_cost.size != 0 else np.array(res1[1].x[counter:counter + T] * res1[2][counter:counter + T], ndmin=2)
                    gasengines_x1 = np.vstack((gasengines_x1,res1[1].x[counter:counter + T])) if gasengines_x1.size != 0 else np.array(res1[1].x[counter:counter + T],ndmin = 2)
                    counter += 4 * T if res1[3] else T
               elif isinstance(e, vo.VPPSolarPanel) or isinstance(e, vo.VPPRenewable):
                    solars_x1_cost = np.vstack((solars_x1_cost, res1[1].x[counter:counter + T] * res1[2][counter:counter + T])) if solars_x1_cost.size != 0 else np.array(res1[1].x[counter:counter + T] * res1[2][counter:counter + T], ndmin=2)
                    solars_x1 = np.vstack((solars_x1,res1[1].x[counter:counter + T])) if solars_x1.size != 0 else np.array(res1[1].x[counter:counter + T],ndmin = 2)
                    counter += T

          # Második optimalizáció eredményeinek feldolgozása
          counter = 0
          for e in i2:
               if isinstance(e, vo.VPPEnergyStorage):
                    storages_x2_cost = np.vstack((storages_x2_cost, np.sum((res2[1].x[counter:counter + 3 * T] * res2[2][counter:counter + 3 * T]).reshape(3,-1), axis=0))) if storages_x2_cost.size != 0 else np.array(np.sum((res2[1].x[counter:counter + 3 * T] * res2[2][counter:counter + 3 * T]).reshape(3,-1), axis=0), ndmin=2)
                    storages_x2 = np.vstack((storages_x2,res2[1].x[counter + 2 * T: counter + 3 * T])) if storages_x2.size != 0 else np.array(res2[1].x[counter + 2 * T: counter + 3 * T],ndmin = 2)
                    counter += 3 * T
               elif isinstance(e, vo.VPPGasEngine):
                    if res2[3]:
                         gasengines_x2_cost = np.vstack((gasengines_x2_cost, np.sum((res2[1].x[counter:counter + 4*T] * res2[2][counter:counter + 4*T]).reshape((T,-1)), axis=1))) if gasengines_x2_cost.size != 0 else np.array(np.sum((res2[1].x[counter:counter + 4*T] * res2[2][counter:counter + 4*T]).reshape((T,-1)), axis=1), ndmin=2)
                    else:
                         gasengines_x2_cost = np.vstack((gasengines_x2_cost, res2[1].x[counter:counter + T] * res2[2][counter:counter + T])) if gasengines_x2_cost.size != 0 else np.array(res2[1].x[counter:counter + T] * res2[2][counter:counter + T], ndmin=2)
                    gasengines_x2 = np.vstack((gasengines_x2,res2[1].x[counter: counter + T])) if gasengines_x2.size != 0 else np.array(res2[1].x[counter: counter + T],ndmin = 2)
                    counter += 4 * T if res2[3] else T
               elif isinstance(e, vo.VPPSolarPanel) or isinstance(e, vo.VPPRenewable):
                    solars_x2_cost = np.vstack((solars_x2_cost, res2[1].x[counter:counter + T] * res2[2][counter:counter + T])) if solars_x2_cost.size != 0 else np.array(res2[1].x[counter:counter + T] * res2[2][counter:counter + T], ndmin=2)
                    solars_x2 = np.vstack((solars_x2,res2[1].x[counter: counter + T])) if solars_x2.size != 0 else np.array(res2[1].x[counter: counter + T],ndmin = 2)
                    counter += T

          # minimális hibák kiszámítása egység típusonként
          ge_mse,ge_diff_mse,ge_ind_mse = pe.min_of_mse(gasengines_x1, gasengines_x2)
          st_mse,st_diff_mse,st_ind_mse = pe.min_of_mse(storages_x1, storages_x2)
          sp_mse,sp_diff_mse,sp_ind_mse = pe.min_of_mse(solars_x1, solars_x2)
          ge_mae,ge_diff_mae,ge_ind_mae = pe.min_of_mae(gasengines_x1, gasengines_x2)
          st_mae,st_diff_mae,st_ind_mae = pe.min_of_mae(storages_x1,storages_x2)
          sp_mae,sp_diff_mae,sp_ind_mae = pe.min_of_mae(solars_x1,solars_x2)


          # Hibák összegzése
          mse_diff_l = list()
          mae_diff_l = list()

          if ge_diff_mse.size != 0:
               mse_diff_l.append(ge_diff_mse)
          if st_diff_mse.size != 0:
               mse_diff_l.append(st_diff_mse)
          if sp_diff_mse.size != 0:
               mse_diff_l.append(sp_diff_mse)

          if ge_diff_mae.size != 0:
               mae_diff_l.append(ge_diff_mae)
          if st_diff_mae.size != 0:
               mae_diff_l.append(st_diff_mae)
          if sp_diff_mae.size != 0:
               mae_diff_l.append(sp_diff_mae)

          mse_diff = np.concatenate(mse_diff_l, axis=0)
          mae_diff = np.concatenate(mae_diff_l, axis=0)

          # Költségek számítása 
          x1_sum_cost_gasengines = np.sum(gasengines_x1_cost)
          x2_sum_cost_gasengines = np.sum(gasengines_x2_cost)
          x1_sum_cost_storages = np.sum(storages_x1_cost)
          x2_sum_cost_storages = np.sum(storages_x2_cost)
          x1_sum_cost_solars = np.sum(solars_x1_cost)
          x2_sum_cost_solars = np.sum(solars_x2_cost)


          # Megjelenítések
          if display_compare and globaldisplay:
               print("Minimális hibák:\nÖsszetett termelés:")
               print("\t\t MSE:", np.average(np.power(mse_diff, 2)))
               print("\t\t MAE:", np.average(np.abs(mae_diff)))
               print("\t\tRMSE:", np.average(np.power(mse_diff, 2))**0.5)
               print("\t\tKöltség összehasonlítása:", x1_sum_cost_gasengines + x1_sum_cost_storages + x1_sum_cost_solars," - ", x2_sum_cost_gasengines + x2_sum_cost_storages + x2_sum_cost_solars)
               print("Gázmotorok:")
               print("\t\t MSE:",ge_mse)
               print("\t\t MAE:",ge_mae)
               print("\t\tRMSE:",ge_mse**0.5)
               print("\t\tKöltség összehasonlítása:", x1_sum_cost_gasengines," - ", x2_sum_cost_gasengines)
               print("Energiatárolók:")
               print("\t\t MSE:",st_mse)
               print("\t\t MAE:",st_mae)
               print("\t\tRMSE:",st_mse**0.5)
               print("\t\tKöltség összehasonlítása:", x1_sum_cost_storages," - ", x2_sum_cost_storages)
               print("Szim. napelemek:")
               print("\t\t MSE:",sp_mse)
               print("\t\t MAE:",sp_mae)
               print("\t\tRMSE:",sp_mse**0.5)
               print("\t\tKöltség összehasonlítása:", x1_sum_cost_solars," - ", x2_sum_cost_solars)
               c.diff_charts(T, {"Gázmotor": ge_diff_mae, "Tároló":st_diff_mae, "Megújuló": sp_diff_mae},
                         {"Gázmotor": ge_ind_mae, "Tároló":st_ind_mae, "Megújuló": sp_ind_mae},
                         disp_part)
          
          if globalsummerize:
               if res1[3]:
                    runtimelst["MILP"].append(res1[4])
                    runtimelst["MILP"].append(res2[4])
                    nanlst["MILP"].append(0)
                    costlstl["MILP"].append(x2_sum_cost_gasengines + x2_sum_cost_storages + x2_sum_cost_solars)

               else:
                    runtimelst["LP"].append(res1[4])
                    runtimelst["LP"].append(res2[4])
                    nanlst["LP"].append(0)
                    costlstl["LP"].append(x2_sum_cost_gasengines + x2_sum_cost_storages + x2_sum_cost_solars)

          return (True, (np.average(np.power(mse_diff, 2)), \
                         np.average(np.abs(mae_diff)), \
                         x1_sum_cost_gasengines + x1_sum_cost_storages + x1_sum_cost_solars, \
                         x2_sum_cost_gasengines + x2_sum_cost_storages + x2_sum_cost_solars),
                    res1[3]
                 )

     else:
          if globalsummerize:
               if res1[3]:
                    nanlst["MILP"].append(1)
               else:
                    nanlst["LP"].append(1)
          if display_compare and globaldisplay:
               print("Az összehasonlítás nem végrehajtható! Az egyik optimalizáció sikertelen volt.")
          return (False, (np.nan, np.nan, np.nan, np.nan), res1[3])
     
# Érzékenység tesztelésére szolgáló függvény
# A második (i2) paramétert egy fun függvényre ceréljük ami megkapja az alapérték változtató érték párot (a,b) és egy százalékot i paraméterül melyekkel az aktuális értéket így számolja: a + i*b
# A további 4 listás paraméter az aktuális teszt input adatait tárolják rendre: az előbb említett függvény első paraméterét növeléshez, csökkentéshez, a fogyasztási görbéket, valamint a d értéket ami még nincs használatban
def sensitivity(i1:list[vo.VPPItem], fun, ulli:list[tuple[float]], llli:list[tuple[float]], T:int=24, lli:list[np.ndarray] = [vo.gen_random_demand(24,0,2000,0)], dli:list[np.ndarray] = [np.zeros(24)], detaild_chart=True, summ_chart=True, debug=False, figsize=(12,9)):
     averagesu = [[],[],[]]
     nof_cant_optu = []
     averagesl = [[],[],[]]
     nof_cant_optl = []
     ismilp = False
     # ha csak 1-1 db van megava ezekből akkor egymás mellé másol annyit, amennyi kell
     if len(ulli) == 1:
          for i in range(len(lli)-1):
               ulli.append(ulli[0])
     if len(llli) == 1:
          for i in range(len(lli)-1):
               llli.append(llli[0])
     if len(dli) == 1:
          for i in range(len(lli)-1):
               dli.append(dli[0])

     # Veégig megyünk az összes fogyasztási görbén és a hozzájuk tartozó input adatokon
     nbr = 1
     for (ul, ll, l, d) in zip(ulli, llli, lli, dli):
          errorsu = np.zeros(0)
          errorsl = np.zeros(0)
          nof_cant_optu.append(0)
          nof_cant_optl.append(0)
          # Lefutattjuk az optimalizációkat a változók csökkentésével/növelésével
          for i in range(5, 101, 5):
               if debug:
                    print(i,". %",sep="")
               # Paraéter növelése tesztek és feldolgozásuk
               i2 = fun(ul, i/100)
               res = test_diff(i1, i2, T, l, d, display_compare = False)
               if not res[0]:
                    nof_cant_optu[-1] += 1
               result = np.array(res[1], ndmin=2)
               errorsu = result if errorsu.size == 0 else np.concatenate([errorsu, result], axis=0)
               
               if debug:
                    print(result)
               # Paraméter csökkentése tesztek és feldolgozásuk
               i2 = fun(ll, -i/100)
               res = test_diff(i1, i2, T, l, d, display_compare = False)
               if not res[0]:
                    nof_cant_optl[-1] += 1
               result = np.array(res[1], ndmin=2)
               errorsl = result if errorsl.size == 0 else np.concatenate([errorsl, result], axis=0)
               if debug:
                    print(result)
               ismilp = res[2]

          # A feldolgozott adatok egymás mellé helyezése a későbbi összegzések okán
          errorsu = np.transpose(errorsu)
          errorsl = np.transpose(errorsl)
               
          averagesu[0].append(errorsu[0,:].reshape(1, -1))
          averagesu[1].append(errorsu[1,:].reshape(1, -1))
          averagesu[2].append((errorsu[3,:]-errorsu[2,:]).reshape(1, -1))

          averagesl[0].append(errorsl[0,:].reshape(1, -1))
          averagesl[1].append(errorsl[1,:].reshape(1, -1))
          averagesl[2].append((errorsl[3,:]-errorsl[2,:]).reshape(1, -1))

          if globalsummerize:
               if ismilp:
                    if not np.isnan(errorsu[1,:]).any():
                         MAElstp["MILP"][globalcurrcat].append(errorsu[1,:])
                    if not np.isnan(errorsu[0,:]).any():
                         RMSElstp["MILP"][globalcurrcat].append(errorsu[0,:]**0.5)
                    if not np.isnan(errorsu[3,:]-errorsu[2,:]).any():
                         MAEcostlstp["MILP"][globalcurrcat].append(errorsu[3,:]-errorsu[2,:])
                    if not np.isnan(errorsl[1,:]).any():
                         MAElstn["MILP"][globalcurrcat].append(errorsl[1,:])
                    if not np.isnan(errorsl[0,:]).any():
                         RMSElstn["MILP"][globalcurrcat].append(errorsl[0,:]**0.5)
                    if not np.isnan(errorsl[3,:]-errorsl[2,:]).any():
                         MAEcostlstn["MILP"][globalcurrcat].append(errorsl[3,:]-errorsl[2,:])
               else:
                    if not np.isnan(errorsu[1,:]).any():
                         MAElstp["LP"][globalcurrcat].append(errorsu[1,:])
                    if not np.isnan(errorsu[0,:]).any():
                         RMSElstp["LP"][globalcurrcat].append(errorsu[0,:]**0.5)
                    if not np.isnan(errorsu[3,:]-errorsu[2,:]).any():
                         MAEcostlstp["LP"][globalcurrcat].append(errorsu[3,:]-errorsu[2,:])
                    if not np.isnan(errorsl[1,:]).any():
                         MAElstn["LP"][globalcurrcat].append(errorsl[1,:])
                    if not np.isnan(errorsl[0,:]).any():
                         RMSElstn["LP"][globalcurrcat].append(errorsl[0,:]**0.5)
                    if not np.isnan(errorsl[3,:]-errorsl[2,:]).any():
                         MAEcostlstn["LP"][globalcurrcat].append(errorsl[3,:]-errorsl[2,:])
          
          # Egy adott teszt megjelenítése
          if detaild_chart and globaldisplay:
               fig, axs = plt.subplots(2, 2, figsize=figsize)
               fig.suptitle(str(nbr) + '. Érzékenységi vizsgálat', fontsize = 30)
               nbr += 1
               for i in range(2):
                    for j in range(2):
                         axs[i, j].ticklabel_format(useOffset=False, style='plain')
                         axs[i, j].set_xlim(0, 95)
                         if j == 1:
                              axs[i, j].set_xlabel("Százalékos változás (" + str(ul[1]) + ")")
                              axs[i, j].set_xticks(np.arange(0, 101, 5),np.arange(0, 101, 5), rotation=40)
                         else:
                              axs[i, j].set_xlabel("Százalékos változás (" + str(ll[1]) + ")")
                              axs[i, j].set_xticks(np.arange(0, 101, 5),np.arange(-100, 1, 5), rotation=40)
                         axs[i, j].set_ylabel("Hiba mértéke")
               axs[0,0].set_title("Érték csökkentése")
               axs[0,1].set_title("Érték növelése")

               axs[0,1].plot(np.arange(0, 101, 5), np.insert(errorsu[0,:]**0.5,0,0), label="MSE Prod")
               axs[0,1].plot(np.arange(0, 101, 5), np.insert(errorsu[1,:],0,0), label="MAE Prod")
               axs[1,1].plot(np.arange(0, 101, 5), np.insert(errorsu[3,:]-errorsu[2,:],0,0), label="Cost")

               max_val = np.abs(np.nanmax([errorsu[0,:]**0.5, errorsu[1,:], np.zeros_like(errorsu[1,:])]))
               min_val = np.abs(np.nanmin([errorsu[0,:]**0.5, errorsu[1,:], np.zeros_like(errorsu[1,:])]))
               delta = np.nanmax([max_val,min_val])

               axs[0,1].set_ylim([-delta - (delta / 10.0) - 10,delta + (delta / 10.0) + 10])

               axs[0,0].plot(np.arange(0, 101, 5), np.flip(np.insert(errorsl[0,:]**0.5,0,0)), label="MSE Prod")
               axs[0,0].plot(np.arange(0, 101, 5), np.flip(np.insert(errorsl[1,:],0,0)), label="MAE Prod")
               axs[1,0].plot(np.arange(0, 101, 5), np.flip(np.insert(errorsl[3,:]-errorsl[2,:],0,0)), label="Cost")


               max_val = np.abs(np.nanmax([errorsl[0,:]**0.5, errorsl[1,:], np.zeros_like(errorsl[1,:])]))
               min_val = np.abs(np.nanmin([errorsl[0,:]**0.5, errorsl[1,:], np.zeros_like(errorsl[1,:])]))
               delta = np.nanmax([max_val,min_val])
               axs[0,0].set_ylim([-delta - (delta / 10.0) - 10,delta + (delta / 10.0) + 10])

               axs[0,0].legend()
               axs[0,1].legend()
               axs[1,0].legend()
               axs[1,1].legend()
               plt.show()
     # Az összes teszt összegzése és maz összegzés vizualizálása
     averagesu[0] = np.vstack(averagesu[0])
     averagesu[1] = np.vstack(averagesu[1])
     averagesu[2] = np.vstack(averagesu[2])


     averagesl[0] = np.vstack(averagesl[0])
     averagesl[1] = np.vstack(averagesl[1])
     averagesl[2] = np.vstack(averagesl[2])

     avgsu = (np.nanmean(averagesu[0], axis=0), np.nanmean(averagesu[1], axis=0), np.nanmean(averagesu[2], axis=0))
     avgsl = (np.nanmean(averagesl[0], axis=0), np.nanmean(averagesl[1], axis=0), np.nanmean(averagesl[2], axis=0))

     averages = (np.nanmean(np.concatenate([averagesu[0], averagesl[0]], axis=0), axis=0), np.nanmean(np.concatenate([averagesu[1], averagesl[1]], axis=0), axis=0), np.nanmean(np.concatenate([averagesu[2], averagesl[2]],axis=0), axis=0))

     if summ_chart and globaldisplay:
          print("Átlagos száma annak, hogy az optimalizáció nem elvégezhető:")
          print("\tÉrték növelésénel:   \t", np.average(nof_cant_optu))
          print("\tÉrték csökkentésénél:\t", np.average(nof_cant_optl))
          print("\tÉrték összesítve:    \t", np.average(nof_cant_optu + nof_cant_optl))


          fig, axs = plt.subplots(2, 2, figsize=figsize)
          fig.suptitle('Érzékenységi vizsgálat összesítve', fontsize = 30)
          for i in range(2):
               for j in range(2):
                    axs[i,j].ticklabel_format(useOffset=False, style='plain')
                    axs[i, j].set_xlim(0, 95)
                    if j == 0:
                         axs[i,j].set_xticks(np.arange(0, 101, 5),np.arange(-100, 1, 5), rotation=40)
                    else:
                         axs[i,j].set_xticks(np.arange(0, 101, 5),np.arange(0, 101, 5), rotation=40)
                    axs[i,j].set_xlabel("Százalékos változás")
                    axs[i,j].set_ylabel("Hiba mértéke")
          axs[0,0].set_title("Érték csökkentése")
          axs[0,1].set_title("Érték növelése")



          axs[0,1].plot(np.arange(0, 101, 5), np.insert(avgsu[0]**0.5,0,0), label="MSE Prod")
          axs[0,1].plot(np.arange(0, 101, 5), np.insert(avgsu[1],0,0), label="MAE Prod")
          axs[1,1].plot(np.arange(0, 101, 5), np.insert(avgsu[2],0,0), label="Cost")

          max_val = np.abs(np.nanmax([avgsu[0]**0.5, avgsu[1], np.zeros_like(avgsu[1])]))
          min_val = np.abs(np.nanmin([avgsu[0]**0.5, avgsu[1], np.zeros_like(avgsu[1])]))
          delta = np.max([max_val,min_val])
          axs[0,1].set_ylim([-delta - (delta / 10.0) - 10,delta + (delta / 10.0) + 10])


          axs[0,0].plot(np.arange(0, 101, 5), np.flip(np.insert(avgsl[0]**0.5,0,0)), label="MSE Prod")
          axs[0,0].plot(np.arange(0, 101, 5), np.flip(np.insert(avgsl[1],0,0)), label="MAE Prod")
          axs[1,0].plot(np.arange(0, 101, 5), np.flip(np.insert(avgsl[2],0,0)), label="Cost")

          max_val = np.abs(np.nanmax([avgsl[0]**0.5, avgsl[1], np.zeros_like(avgsl[1])]))
          min_val = np.abs(np.nanmin([avgsl[0]**0.5, avgsl[1], np.zeros_like(avgsl[1])]))
          delta = np.max([max_val,min_val])
          axs[0,0].set_ylim([-delta - (delta / 10.0) - 10,delta + (delta / 10.0) + 10])

          axs[0,0].legend()
          axs[0,1].legend()
          axs[1,0].legend()
          axs[1,1].legend()
          plt.show()

          fig, axs = plt.subplots(2, 1, figsize=figsize)
          axs[0].plot(np.arange(0, 101, 5), np.insert(averages[0]**0.5,0,0), label="MSE Prod")
          axs[0].plot(np.arange(0, 101, 5), np.insert(averages[1],0,0), label="MAE Prod")
          axs[1].plot(np.arange(0, 101, 5), np.insert(averages[2],0,0), label="Cost")

          max_val = np.abs(np.nanmax([averages[0]**0.5, averages[1], np.zeros_like(averages[1])]))
          min_val = np.abs(np.nanmin([averages[0]**0.5, averages[1], np.zeros_like(averages[1])]))
          delta = np.max([max_val,min_val])
          axs[0].set_ylim([-delta - (delta / 10.0) - 10,delta + (delta / 10.0) + 10])

          for j in range(2):
               axs[j].set_xlim(0, 95)
               axs[j].ticklabel_format(useOffset=False, style='plain')
               axs[j].set_xticks(np.arange(0, 100, 5),np.arange(0, 100, 5), rotation=40)
               axs[j].set_xlabel("Százalékos változás")
               axs[j].set_ylabel("Hiba mértéke")
          axs[0].set_title("Minden irányú összesítés")

          axs[0].legend()
          axs[1].legend()
          plt.show()
     
     return avgsu, avgsl, averages, np.average(nof_cant_optu), np.average(nof_cant_optl), np.average(nof_cant_optu + nof_cant_optl)

def sensitivity_2(i1:list[vo.VPPItem], fun, variable_names, slli:list[tuple[float]], T:int=24, lli:list[np.ndarray] = [vo.gen_random_demand(24,0,2000,0)], dli:list[np.ndarray] = [np.zeros(24)], detaild_chart=True, summ_chart=True, debug=False, figsize=(12,9), view=[[(30,210)]]):

     if len(slli) == 1:
          for i in range(len(lli)-1):
               slli.append(slli[0])
     if len(dli) == 1:
          for i in range(len(lli)-1):
               dli.append(dli[0])
     if len(view) == 1:
          if len(view[0]) == 1:
               view[0].append(view[0][0])
               view[0].append(view[0][0])
          for i in range(len(lli)-1):
               view.append(view[0])
     else:
          for i in range(len(view)):
               if len(view[i]) == 1:
                    view[i].append(view[i][0])
                    view[i].append(view[i][0])

     for (sl, l, d, ind) in zip(slli, lli, dli, range(0,len(lli))):
          allerrors = list()
          for i in range(-100, 101, 5):
               errors = np.zeros(0)
               for j in range(-100, 101, 5):
                    i2 = fun(sl, i/100, j/100)
                    res = test_diff(i1, i2, T, l, d, display_compare = False)
                    result = np.array(res[1], ndmin=2)
                    errors = result if errors.size == 0 else np.concatenate([errors, result], axis=0)
               errors = errors.transpose()
               errors[2,:] = errors[3,:] - errors[2,:]
               errors = errors[:3,:]
               allerrors.append(np.array(errors, ndmin=3))
          allerrors = np.vstack(allerrors)
          if globaldisplay:
               fig, ax = plt.subplots(3, 1, subplot_kw=dict(projection='3d'),figsize=(10,30) )
               x, y = np.meshgrid(np.arange(-100, 101, 5), np.arange(-100, 101, 5))
               ax[0].set_title("RMSE Prod")
               ax[1].set_title("MAE Prod")
               ax[2].set_title("MAE Cost")
               for i in range(0,3):
                    if i == 0:
                         vals = allerrors[:,i,:]**0.5
                    else:
                         vals = allerrors[:,i,:]
                    max_val = np.nanmax([vals, np.zeros_like(vals)])
                    min_val = np.abs(np.nanmin([vals, np.zeros_like(vals)]))
                    ax[i].set_zlim((-min_val - (min_val / 10.0) - 10,max_val + (max_val / 10.0) + 10))
                    img = ax[i].plot_surface(y, x, vals, cmap='BrBG')
                    fig.colorbar(img, ax=ax[i])
                    ax[i].view_init(elev=view[ind][i][0], azim=view[ind][i][1])
                    ax[i].set_xlabel("A(z) " + variable_names[0] + " paraméter %-os változása")
                    ax[i].set_ylabel("A(z) " + variable_names[1] + " paraméter %-os változása")
               plt.show()


def sensitivity_by_participants(fun, var_from, var_to, density, l, d, T, view=None, addcontour=False):
     gc = 1
     tc = 1
     sc = 1
     labels = list()

     xx = np.linspace(var_from, var_to, density, endpoint=True)
     thearray = list()
     for x in xx:
          items = fun(x)
          result = vo.solve(items, T, l, d, display=False, return_res=True)
          if not result[0]:
               thearray.append(np.array(np.zeros((len(items), T)), ndmin=3))
          else:
               tmp = list()
               counter = 0
               for e in items:
                    if isinstance(e, vo.VPPEnergyStorage):
                         tmp.append(np.array([-1*result[1].x[counter+T:counter+ 2*T],result[1].x[counter+2*T:counter+ 3*T]]))
                         counter += 3 * T
                         if x == xx[0]:
                              labels.append(str(tc) + ". Tároló töltése")
                              labels.append(str(tc) + ". Tároló leadása")
                              tc += 1
                    elif isinstance(e, vo.VPPGasEngine):
                         tmp.append(np.array(result[1].x[counter: counter + T],ndmin = 2))
                         counter += 4 * T if result[3] else T
                         if x == xx[0]:
                              labels.append(str(gc) + ". Gázmotor termelése")
                              gc += 1
                    elif isinstance(e, vo.VPPSolarPanel) or isinstance(e, vo.VPPRenewable):
                         tmp.append(np.array(result[1].x[counter: counter + T],ndmin = 2))
                         counter += T
                         if x == xx[0]:
                              labels.append(str(sc) + ". Napelem termelése")
                              sc += 1
               tmp = np.array(np.vstack(tmp), ndmin=3)
               thearray.append(tmp)
     thearray = np.vstack(thearray)
     yy = np.arange(1,25)
     x, y = np.meshgrid(xx, yy)
     colors = ["b", "r", "g", "purple", "yellow"]
     if globaldisplay:
          fig, ax = plt.subplots(subplot_kw=dict(projection='3d'),figsize=(16, 12))
          for (label, i, c) in zip(labels, range(len(labels)), colors):
               ax.plot_wireframe(x, y, thearray[:,i,:].transpose(), label=label, color=c, rcount=0, ccount=density)
               if addcontour:
                    ax.contour(x, y, thearray[:,i,:].transpose(), zdir='x', offset=var_from-1)
          ax.set_ylabel("Időintervallum")
          ax.set_xlabel("Paraméter értéke")
          ax.set_zlabel("Termelt energia")
          ax.legend()
          if view is not None:
               ax.view_init(elev=view[0], azim=view[1])
          plt.show()

def summerize():
     print("=================================================")
     print("Futási idők átlaga:")
     print("LP: " + str(np.nanmean(runtimelst["LP"])))
     print("MILP: " + str(np.nanmean(runtimelst["MILP"])))
     print("=================================================")
     print("Nem optimalizálható esetek számának átlaga:")
     print("LP: " + str(np.nanmean(nanlst["LP"])))
     print("MILP: " + str(np.nanmean(nanlst["MILP"])))
     print("=================================================")
     print("Átlagos költség:")
     print("LP: " + str(np.nanmean(costlstl["LP"])))
     print("MILP: " + str(np.nanmean(costlstl["MILP"])))
     print("=================================================")
     print("Átlagos MAE termelés hiba:")
     summerizeplt(MAElstp, MAElstn)
     print("=================================================")
     print("Átlagos RMAE termelés hiba:")
     summerizeplt(RMSElstp, RMSElstn)
     print("=================================================")
     print("Átlagos költség hiba:")
     summerizeplt(MAEcostlstp, MAEcostlstn)

def summerizeplt(datap, datam, figsize=(20,15)):
     for k in range(4):
          fig, axs = plt.subplots(2, 2, figsize=(20,15))
          for i in range(2):
               for j in range(2):
                    axs[i,j].ticklabel_format(useOffset=False, style='plain')
                    axs[i,j].set_xlim(0, 95)
                    if j == 0:
                         axs[i,j].set_xticks(np.arange(0, 101, 5),np.arange(-100, 1, 5), rotation=40)
                    else:
                         axs[i,j].set_xticks(np.arange(0, 101, 5),np.arange(0, 101, 5), rotation=40)
                    axs[i,j].set_xlabel("Százalékos változás")
                    axs[i,j].set_ylabel("Hiba mértéke")
          axs[0,0].set_title("Érték csökkentése LP")
          axs[0,1].set_title("Érték növelése LP")
          axs[1,0].set_title("Érték csökkentése  MILP")
          axs[1,1].set_title("Érték növelése MILP")

          axs[0,1].plot(np.arange(0, 101, 5), np.insert(np.nanmean(np.vstack(datap["LP"][k]), axis=0),0,0))
          axs[0,0].plot(np.arange(0, 101, 5), np.flip(np.insert(np.nanmean(np.vstack(datam["LP"][k]), axis=0),0,0)))
          axs[1,1].plot(np.arange(0, 101, 5), np.insert(np.nanmean(np.vstack(datap["MILP"][k]), axis=0),0,0))
          axs[1,0].plot(np.arange(0, 101, 5), np.flip(np.insert(np.nanmean(np.vstack(datam["MILP"][k]), axis=0),0,0)))
          plt.show()