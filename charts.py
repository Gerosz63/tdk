import numpy as np
import vppopt as vo
import matplotlib.pyplot as plt

def chart(time, items, costs, result, l, d, detaild=False, detaildItems=[], ismilp=False, main_title:str=""):
    if detaild:
        fig, axs = plt.subplots(2, 2, figsize=(20,15))

        # Az értékeke kiírása rövidítés nélkül
        axs[0,0].ticklabel_format(useOffset=False, style='plain')
        axs[1,0].ticklabel_format(useOffset=False, style='plain')
        axs[0,1].ticklabel_format(useOffset=False, style='plain')
        axs[1,1].ticklabel_format(useOffset=False, style='plain')

        production_chart(time, items, result, l + d, axs[0, 0], ismilp)
        production_chart_by_participants(time, items, result,  axs[0, 1], detaildItems, ismilp)
        cost_chart(time, items, costs, result,  axs[1, 0], ismilp)
        cost_chart_participants(time, items, costs, result,  axs[1, 1], detaildItems, ismilp)
        if main_title != "":
            fig.suptitle(main_title, fontsize=30)
    else:
        fig, axs = plt.subplots(2, figsize=(20,15))

        # Az értékeke kiírása rövidítés nélkül
        axs[0].ticklabel_format(useOffset=False, style='plain')
        axs[1].ticklabel_format(useOffset=False, style='plain')

        production_chart(time, items, result, l + d, axs[0], ismilp)  
        cost_chart(time, items, costs, result,  axs[1], ismilp)
        if main_title != "":
            fig.suptitle(main_title, fontsize=30)

    plt.show()


def cost_chart(time, items, cost, resultx, ax, ismilp):
    # Kategóriák létrehozása
    stc = np.zeros(time,dtype=np.float32)
    ec = np.zeros(time,dtype=np.float32)
    sc = np.zeros(time, dtype=np.float32)
    sp = np.zeros(time, dtype=np.float32)
    wp = np.zeros(time, dtype=np.float32)
    counter = 0

    for item in items:
        if isinstance(item, vo.VPPEnergyStorage):
            counter += time
            stc += cost[counter:counter + time] * resultx[counter:counter + time] + cost[counter + time:counter + 2 * time] * resultx[counter + time:counter + 2 * time]
            counter += 2 * time
        elif isinstance(item, vo.VPPGasEngine):
            ec += cost[counter:counter + time] * resultx[counter:counter + time]
            if ismilp:
                ec += cost[counter + time:counter + 2 * time] * resultx[counter + time:counter + 2 * time] + cost[counter + 2 * time:counter + 3 * time] * resultx[counter + 2 * time:counter + 3 * time]
                counter += 3 * time
            counter += time
        elif isinstance(item, vo.VPPSolarPanel):
            sc += cost[counter:counter + time] * resultx[counter:counter + time]
            counter += time
        elif isinstance(item, vo.VPPRenewable):
            if item.label == "SP":
                sp += cost[counter:counter + time] * resultx[counter:counter + time]
                counter += time
            elif item.label == "WP":
                wp += cost[counter:counter + time] * resultx[counter:counter + time]
                counter += time


    # Első elem duplikálása, normális megjelenítés érdekében
    stc = np.insert(stc, 0, stc[0])
    ec = np.insert(ec, 0, ec[0])
    sc = np.insert(sc, 0, sc[0])
    sp = np.insert(sp, 0, sp[0])
    wp = np.insert(wp, 0, wp[0])
    x = np.arange(0, time+1, 1)
    
    # Y tengely maximumának meghatározása
    summa = sc + ec + stc + sp + wp
    ymax = np.max(summa)
    ax.set_ylim([0.0, ymax + (ymax / 10.0)])

    # Rajzolás
    ax.step(x, sc + stc + ec + sp + wp, color="lightblue", label="Szélerőművek költsége")
    ax.step(x, sc + stc + ec + sp, color="yellow", label="Napelemek költsége")
    ax.step(x, sc + stc + ec, color="orange", label="Energia tárolók költsége")
    ax.step(x, sc + ec, color="green", label="Szimulált napelemek költsége")
    ax.step(x, ec, color="blue", label="Gázmotorok költsége")

    ax.fill_between(x, ec, color="blue", step="pre", alpha=0.25)
    ax.fill_between(x, ec, sc + ec, color="green", step="pre", alpha=0.25)
    ax.fill_between(x, sc + ec, sc + stc + ec, color="orange", step="pre", alpha=0.25)
    ax.fill_between(x, sc + stc + ec, sc + stc + ec + sp, color="yellow", step="pre", alpha=0.25)
    ax.fill_between(x, sc + stc + ec + sp, sc + stc + ec + sp + wp, color="lightblue", step="pre", alpha=0.25)

    ax.legend()
    ax.set_xticks(x)
    ax.set_xlabel("Idő intervallumok")
    ax.set_ylabel("Költség")
    ax.set_title("Költség összesítve")

def production_chart(time:int, items:list[vo.VPPItem], resultx:list[float], L:np.ndarray, ax, ismilp):
    # Kategóriák létrehozása
    engines = np.zeros(time,dtype=np.float32)
    solars = np.zeros(time,dtype=np.float32)
    stores = np.zeros(time,dtype=np.float32)
    sp = np.zeros(time, dtype=np.float32)
    wp = np.zeros(time, dtype=np.float32)

    counter = 0
    for item in items:
        if isinstance(item, vo.VPPEnergyStorage):
            stores += resultx[counter + 2 * time:counter + 3 * time]       
            counter += 3 * time
        elif isinstance(item, vo.VPPGasEngine):
                engines += resultx[counter:counter + time]
                counter += 4 * time if ismilp else time
        elif isinstance(item, vo.VPPSolarPanel):
                solars += resultx[counter:counter + time]
                counter += time
        elif isinstance(item, vo.VPPRenewable):
            if item.label == "SP":
                sp += resultx[counter:counter + time]
                counter += time
            elif item.label == "WP":
                wp += resultx[counter:counter + time]
                counter += time

    # Első elem duplikálása, normális megjelenítés érdekében
    engines = np.insert(engines, 0, engines[0])
    solars = np.insert(solars, 0, solars[0])
    stores = np.insert(stores, 0, stores[0])
    sp = np.insert(sp, 0, sp[0])
    wp = np.insert(wp, 0, wp[0])

    L = np.insert(L, 0, L[0])
    x = np.arange(0, time + 1, 1)

    # Y tengely maximumának meghatározása
    ymax = np.max([L, sp + wp + stores + solars + engines])
    ax.set_ylim([0.0, ymax + (ymax / 10.0)])

    # Kirajzolás
    ax.step(x, solars + engines + stores + sp + wp, color="lightblue", label="Szélerőművek")
    ax.step(x, solars + engines + stores + sp, color="yellow", label="Napelemek")
    ax.step(x, solars + engines + stores, color="orange", label="Tárolók")
    ax.step(x, solars + engines, color="green", label="Szimulált napelemek")
    ax.step(x, engines, color="blue", label="Gázmotorok",where="pre")

    ax.step(x, L, color="red", label="Várt fogyasztás", linestyle=":", linewidth=2)
    ax.legend()
    ax.set_xticks(x)

    ax.fill_between(x, engines, color="blue", step="pre", alpha=0.25)
    ax.fill_between(x, engines, solars + engines, color="green", step="pre", alpha=0.25)
    ax.fill_between(x, solars + engines, solars + engines + stores, color="orange", step="pre", alpha=0.25)
    ax.fill_between(x, solars + engines + stores, solars + engines + stores + sp, color="yellow", step="pre", alpha=0.25)
    ax.fill_between(x, solars + engines + stores + sp, solars + engines + stores + sp + wp, color="lightblue", step="pre", alpha=0.25)
    ax.set_xlabel("Idő intervallumok")
    ax.set_ylabel("Termelt energia")
    ax.set_title("Termelés összesítve")


def cost_chart_participants(time, items, cost, resultx, axs, itemstodisplay, ismilp):
    counter = 0
    ymax = 0
    gec = 1
    strc = 1
    sspc = 1
    spc = 1
    wpc = 1
    x = np.arange(0, time + 1, 1)

    for i in items:
        if i not in itemstodisplay:
            if isinstance(i,vo.VPPGasEngine):
                counter += 4 * time if ismilp else time
            elif isinstance(i,vo.VPPSolarPanel):
                counter += time
            elif isinstance(i, vo.VPPEnergyStorage):
                counter += 3 * time
            continue
        
        if isinstance(i,vo.VPPGasEngine):
            if ismilp:
                item = resultx[counter:counter + time] * cost[counter:counter + time] + resultx[counter + time:counter + 2*time] * cost[counter + time:counter + 2*time] + resultx[counter + 2*time:counter + 3*time] * cost[counter + 2*time:counter + 3*time]
            else:
                item = resultx[counter:counter + time] * cost[counter:counter + time]
            counter += 4 * time if ismilp else time
            label = "Gázmotor "+ str(gec)
            gec += 1
            
        elif isinstance(i, vo.VPPEnergyStorage):
            counter += time
            item = resultx[counter:counter + time] * cost[counter:counter + time] + resultx[counter + time:counter + 2 * time] * cost[counter + time:counter + 2 * time]
            counter += 2 * time
            label = "Tároló "+ str(strc)
            strc += 1
            
        elif isinstance(i, vo.VPPSolarPanel):
            item = resultx[counter:counter + time] * cost[counter:counter + time]
            counter += time
            label = "Szim. Napelem " + str(sspc)
            sspc += 1
        elif isinstance(i, vo.VPPRenewable):
            if i.label == "SP":
                item = resultx[counter:counter + time] * cost[counter:counter + time]
                counter += time
                label = "Napelem " + str(spc)
                spc += 1
            elif i.label == "WP":
                item = resultx[counter:counter + time] * cost[counter:counter + time]
                counter += time
                label = "Szélerőmű " + str(wpc)
                wpc += 1
        else:
            item = np.zeros(1)
        ymax = np.max([ymax, np.max(item)])    
        item = np.insert(item, 0, item[0])
        axs.step(x, item, label=label)

    axs.set_ylim([0.0, ymax + (ymax / 10.0)])
    axs.legend()
    axs.set_xticks(x)
    axs.set_xlabel("Idő intervallumok")
    axs.set_ylabel("Költség")
    axs.set_title("Résztvevők költségei")

def production_chart_by_participants(time, items, resultx, axs, itemstodisplay, ismilp):
    counter = 0
    ymax = 0
    gec = 1
    strc = 1
    sspc = 1
    spc = 1
    wpc = 1
    x = np.arange(0, time + 1, 1)

    for i in items:
        if i not in itemstodisplay:
            if isinstance(i,vo.VPPGasEngine):
                counter += 4 * time if ismilp else time
            elif isinstance(i,vo.VPPSolarPanel):
                counter += time
            elif isinstance(i, vo.VPPEnergyStorage):
                counter += 3*time
            continue
        if isinstance(i, vo.VPPGasEngine):
            item = resultx[counter:counter + time]
            counter += 4 * time if ismilp else time
            label = "Gázmotor "+ str(gec)
            gec += 1
            
        elif isinstance(i, vo.VPPEnergyStorage):
            item = resultx[counter:counter + time]
            counter += time
            charge = resultx[counter:counter + time]
            counter += time
            discharge = resultx[counter:counter + time]
            counter += time

            charge = np.insert(charge, 0, charge[0])
            discharge = np.insert(discharge, 0, discharge[0])

            ymax = np.max([ymax, np.max([charge,discharge])])

            label = "Tároló " + str(strc)

            axs.step(x, charge, label=label + " töltés")
            axs.step(x, discharge, label=label + " leadás")
            strc += 1

        elif isinstance(i, vo.VPPSolarPanel):
            item = resultx[counter:counter + time] 
            solarmax = np.copy(i.r_max_values)
            counter += time
            label = "Szim. Napelem " + str(sspc)
            solarmax = np.insert(solarmax, 0, solarmax[0])
            ymax = np.max([ymax, np.max(solarmax)])
            axs.step(x, solarmax, label=label + " max")
            sspc += 1
        elif isinstance(i, vo.VPPRenewable):
            if i.label == "SP":
                item = resultx[counter:counter + time] 
                counter += time
                label = "Napelem " + str(spc)
                spc += 1
            elif i.label == "WP":
                item = resultx[counter:counter + time] 
                counter += time
                label = "Szélerőmű " + str(spc)
                wpc += 1
        else:
            item = np.zeros(1)
        ymax = np.max([ymax, np.max(item)])
        item = np.insert(item, 0, item[0])
        axs.step(x, item, label=label)
        
    axs.set_ylim([0.0, ymax + (ymax / 10.0)])
    axs.legend()
    axs.set_xticks(x)
    axs.set_xlabel("Idő intervallumok")
    axs.set_ylabel("Termelt energia")
    axs.set_title("Termelés résztvevőnként")

def diff_charts(time:int,resx:dict, x_ind:dict, disp_part:dict[str,list[int]]|None=None):
    if disp_part is None:
        disp_part = {"Tároló": range(resx["Tároló"].shape[0]), "Gázmotor": range(resx["Gázmotor"].shape[0]), "Megújuló":range(resx["Megújuló"].shape[0])}

    
    # Első elem duplikálása, normális megjelenítés érdekében


    diff_st_prod = np.sum((resx["Tároló"] if resx["Tároló"].size != 0 else np.zeros((1, time))), axis=0)
    diff_e_prod = np.sum((resx["Gázmotor"] if resx["Gázmotor"].size != 0 else np.zeros((1, time))), axis=0)
    diff_s_prod = np.sum((resx["Megújuló"] if resx["Megújuló"].size != 0 else np.zeros((1, time))), axis=0)
    all_diff_prod = np.sum(np.concatenate([diff_st_prod.reshape((1,-1)), diff_e_prod.reshape((1,-1)), diff_s_prod.reshape((1,-1))], axis=0), axis=0)

    diff_st_prod = np.insert(diff_st_prod, 0, diff_st_prod[0])
    diff_e_prod = np.insert(diff_e_prod, 0, diff_e_prod[0])
    diff_s_prod = np.insert(diff_s_prod, 0, diff_s_prod[0])
    all_diff_prod = np.insert(all_diff_prod, 0, all_diff_prod[0])

    x = np.arange(0,time + 1,1)

    # Ábrázoljuk a költségbeli eltéréseket
    fig, axs = plt.subplots(2, figsize=(20,15))
    fig.suptitle('Összesített különbségek', fontsize = 30)
    for i in range(2):
        axs[i].ticklabel_format(useOffset=False, style='plain')
        axs[i].set_xticks(x)
        axs[i].set_xlabel("Idő intervallumok")
        axs[i].set_ylabel("Különbségek")

    axs[0].set_title("Termelések")
    axs[1].set_title("Részvevő termelések")

    # Ábrázoljuk a termelésbeli eltéréseket
    unit_diff_charts(x,diff_s_prod,axs[0],'Eltérések a szim. napelemek termelésében','green')
    unit_diff_charts(x,diff_st_prod,axs[0],'Eltérések az energiatárolók termelésében','orange', diff_s_prod)
    unit_diff_charts(x,diff_e_prod,axs[0],'Eltérések a gázmotorok termelésében','blue', diff_st_prod + diff_s_prod)
    unit_diff_charts(x,all_diff_prod,axs[0],'Eltérések az összesített termelésben','red', None, False)
    max_val = np.abs(np.max([all_diff_prod, diff_e_prod, diff_st_prod, diff_s_prod]))
    min_val = np.abs(np.min([all_diff_prod, diff_e_prod, diff_st_prod, diff_s_prod]))
    delta = np.max([max_val,min_val])
    axs[0].set_ylim([-delta - (delta / 10.0) - 10,delta + (delta / 10.0) + 10])


    


    for i in disp_part["Tároló"]:
        unit_diff_charts(x, repet_first(resx["Tároló"][i]), axs[1],"Tároló " +str(x_ind["Tároló"][i] + 1) + " - " + str(i + 1))
    for i in disp_part["Gázmotor"]:
        unit_diff_charts(x, repet_first(resx["Gázmotor"][i]), axs[1],"Gázmotor " +str(x_ind["Gázmotor"][i] + 1) + " - " + str(i + 1))
    for i in disp_part["Megújuló"]:
        unit_diff_charts(x, repet_first(resx["Megújuló"][i]), axs[1],"Megújuló " +str(x_ind["Megújuló"][i] + 1) + " - " + str(i + 1))
    
    axs[0].legend()
    axs[1].legend()
    plt.show()
    
def unit_diff_charts(x:np.ndarray,values:np.ndarray,axs,title:str,clr:str|None = None, sfrom:np.ndarray|None=None, fill=True):
    if sfrom is None:
        sfrom = np.zeros_like(x)
    if fill:
        if clr is None:
            axs.step(x,values, label=title)
            axs.fill_between(x,sfrom,values,step = 'pre',alpha = 0.25)
        else:
            axs.step(x,values,color=clr, label=title)
            axs.fill_between(x,sfrom,values,step = 'pre',color = clr,alpha = 0.25)
    else:
        if clr is None:
            axs.step(x,values, label=title, linestyle=":", linewidth=2)
        else:
            axs.step(x,values,color=clr, label=title, linestyle=":", linewidth=2)

def repet_first(x:np.ndarray):
    return np.insert(x, 0, x[0])