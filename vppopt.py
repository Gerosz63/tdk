import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint
import math
from scipy.linalg import toeplitz
import time

class VPPItem:
    def __init__(self) -> None:
        self.is_MILP = False
    def lp_get_constraints(self, T:int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return np.zeros((1,T)), np.zeros(1), np.zeros((1,T)), np.zeros(1) # leq mm, leq v, eq m, eq v 
    def lp_get_cost(self, T:int) -> np.ndarray:
        return np.zeros(T)
    def milp_get_constraints(self, T:int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.zeros(1), np.zeros((1,T)), np.zeros(1) # lb, m, ub
    def milp_get_cost(self, T:int) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(T), np.zeros(T) # cf, itr
    
    
class VPPGasEngine(VPPItem):
    def __init__(self, g_max:float, g_plus_max:float, g_minus_max:float, cost:float, g0:float|None = None, #LP
                 g_min:float|None = None, startcost:float|None = None, endcost:float|None = None, min_on:int|None = None, min_off:int|None = None, min0_on:int|None = None, min0_off:int|None = None,
                 st:int|None = None, et:int|None = None, st0:int|None = None, et0:int|None = None, sts:int|None = None,
                 go0:int|None=None):
        # Ellenőrzések:
        if g_max < 0. or g_plus_max < 0. or g_minus_max < 0.:
            raise Exception("A g_max, g_plus_max, g_minus_max értékek nem lehetnek 0-nál kisebb értékek!")
        
        self.g_max = g_max
        self.g_plus_max = g_plus_max
        self.g_minus_max = g_minus_max
        self.cost = cost

        #Amennyiben a g0 nem megfelelő értékű javítjuk azt
        if g0 != None:
            if g0 < 0:
                g0 = 0
            elif g0 > g_max:
                g0 = g_max
        self.g0 = g0
        if g_min is not None or startcost is not None or endcost is not None or min_on is not None or min_off is not None or min0_on is not None or min0_off is not None:
            self.is_MILP = True
        else:
            self.is_MILP = False
        
        # Gázturbina esetén is fennállnak a milpes kényszerek
        isTurbine = False
        if st is not None or et is not None or st0 is not None or et0 is not None or sts is not None:
            self.is_MILP = True
            isTurbine = True

        # Hogyha milp akkor beállítjuk a milpes mezőket
        if self.is_MILP:

            self.g_min = g_min if g_min is not None else 0
            if self.g0 != None and self.g0 < self.g_min:
                self.g0 = 0
            self.startcost = startcost if startcost is not None else 0
            self.endcost = endcost if endcost is not None else 0
            if go0 == None:
                self.go0 = 0 if g0 == 0 else 1
            else: 
                self.go0 = go0
            self.min_on = min_on if min_on is not None else 3
            self.min_off = min_off if min_off is not None else 3
            self.min0_on = min0_on if min0_on is not None else self.min_on
            self.min0_off = min0_off if min0_off is not None else self.min_off
            self.st = 0
            self.et = 0
            self.st0 = 0
            self.et0 = 0

            if self.min0_on > self.min_on:
                self.min0_on = self.min_on
            if self.min0_off > self.min_off:
                self.min0_off = self.min_off
            
            if (self.go0 == 0 and self.g0 != 0) or (self.go0 == 1 and self.g0 == 0 and self.g_min != 0):
                raise Exception("Ellentmondás a g0 és go0 értékek között!")
            if self.min_on < 1 or self.min_off < 1:
                raise Exception("A min_on és a min_off értékeknek pozitívnak kell lenniük!")
            if self.min0_on < 0 or self.min0_off < 0:
                raise Exception("A min0_on és a min0_off értékek nem lehetnek negatívak!")
            
            if self.min0_on < self.min_on and self.min0_off < self.min_off:
                raise Exception("A min0_on és min0_off értékek nem lehetnek egyszerre kisebbek, mint min_on és min_off értékei.")

            
            if self.g_min < 0:
                raise Exception("A g_min érték nem lehetnek negatívak!")  
            # A gázturbinához kapcsolódó rész
            if isTurbine:
                self.st = st if st is not None else 1
                self.et = et if et is not None else 1
                self.st0 = st0 if st0 is not None else self.st
                self.et0 = et0 if et0 is not None else self.et
                self.sts = sts if sts is not None else 0
                
                if self.st0 > self.st:
                    self.st0 = self.st
                if self.et0 > self.et:
                    self.et0 = self.et
                
                if self.st < 0 or self.et < 0 or self.st0 < 0 or self.et0 < 0 or self.sts < 0 or self.sts > 3:
                    raise Exception("Az st, az et, az st0 és az et0 értékek nem lehetnek negatívak és az sts értéke 0 és 3 között kell lennie!")
                if self.et0 < self.et and self.st0 < self.st:
                    raise Exception("A et0 és st0 értékek nem lehetnek egyszerre kisebbek, mint et és st értékei.")

    def lp_get_constraints(self, T:int) -> tuple[np.ndarray, np.ndarray, np.ndarray|None, np.ndarray|None]:
        mm, vm = LinearConstraints.generate_constraints_min_max(T, 0, self.g_max)
        mmc, vmc = LinearConstraints.generate_constraints_min_max_time(T, self.g_minus_max, self.g_plus_max)

        
        if self.g0 != None:
            s0m = np.zeros((2, T), dtype=np.float32)
            s0m[0, 0] = -1.
            s0m[1, 0] = 1
            s0v = np.array([self.g_minus_max - self.g0, self.g_plus_max + self.g0], dtype=np.float32)
            m, v = MatrixMerger.Merge_LP_Constraints([mm, mmc, s0m], [vm, vmc, s0v])
        else:
            m, v = MatrixMerger.Merge_LP_Constraints([mm, mmc], [vm, vmc])


        return m, v, None, None
    def lp_get_cost(self, T:int) -> np.ndarray:
        return np.full(T, self.cost, dtype=np.float32)
    
    def milp_get_constraints(self, T: int): # Elkészíti a MILP-hez szükséges kényszereket (jelölések: u/a = alsó/felső korlátokat tartalmazó vektor; m = az együtthatókat tartalmazó mátrix)
        # A korlátosságra vonatkozó kényszerek megvalósítása
        limit_u_work = np.zeros(2*T,dtype=np.float32)
        limit_u_work[:T] = -np.inf
        limit_m1_work = np.concatenate([np.eye(T,T,dtype=np.float32),np.zeros((T,2*T),dtype=np.float32),np.eye(T,T,dtype=np.float32)*(-self.g_max)],axis=1)
        limit_m2_work = np.concatenate([np.eye(T,T,dtype=np.float32),np.zeros((T,2*T),dtype=np.float32),np.eye(T,T,dtype=np.float32)*(-self.g_min)],axis=1)
        limit_a_work = np.zeros(2*T,dtype=np.float32)
        limit_a_work[T:] = np.inf
        limit_u_start,limit_m_start,limit_a_start = MILPConstraints.generate_constraints_for_limitation(T,1.)
        limit_u_end,limit_m_end,limit_a_end = MILPConstraints.generate_constraints_for_limitation(T,1.)
        limit_u_on,limit_m_on,limit_a_on = MILPConstraints.generate_constraints_for_limitation(T,1.)
        limit_u,limit_m,limit_a = MatrixMerger.MergeDiff_MILP_Constraints([limit_u_start,limit_u_end,limit_u_on],[limit_m_start,limit_m_end,limit_m_on],[limit_a_start,limit_a_end,limit_a_on])
        limit_m = np.concatenate([np.zeros((3*T,T),dtype=np.float32),limit_m],axis=1)

        # A korlátos szabályozhatóságra vonatkozó kényszerek megvalósítása
        control_u1,control_m1,control_a1 = MILPConstraints.generate_constraints_to_controllability(T,-np.inf,0,4)
        control_u2,control_m2,control_a2 = MILPConstraints.generate_constraints_to_controllability(T,0,np.inf,4)
        control_m2 = np.concatenate([control_m2[:,:T],np.zeros((T-1,T),dtype=np.float32), np.eye(T-1,T,1) * self.g_min, np.eye(T-1,T)*(self.g_minus_max)],axis=1)

        if self.st > 0: # Ha turbináról van szó
            control_m1 = np.concatenate([control_m1[:,:T],np.eye(T-1,T)*(-self.g_min),np.zeros((T-1,T),dtype=np.float32),np.eye(T-1,T,1)*(-self.g_plus_max)],axis=1)
            # Az állapotváltozásokra vonatkozó kényszerek megvalósítása
            sum_status_u = np.zeros(T, dtype=np.float32)
            sum_status_m = np.concatenate([np.zeros((T,T), np.float32), np.eye(T,T),np.eye(T,T),np.eye(T,T)], axis=1)
            sum_status_a = np.full(T, 1, dtype=np.float32)
            sum_m = MILPConstraints.generate_delta_matrix(T,T-1, 1)
            status_u = np.full(T-1,-1,dtype=np.float32)
            status_m = np.concatenate([np.zeros((T-1,T), np.float32), sum_m, sum_m, sum_m], axis=1)
            status_a = np.full(T-1,1,dtype=np.float32)

            # A kezdeti állapotváltozásokra vonatkozó kényszerek megvalósítása
            if self.sts == 0: # 0. pillanatban inaktív a motor
                s_status_u = np.full(2,0,dtype=np.float32)
                s_status_m1 = np.concatenate([np.zeros((1,T), np.float32), np.eye(1, T, dtype=np.float32), np.zeros((1,T), np.float32), np.zeros((1,T), np.float32)], axis=1)
                s_status_m2 = np.concatenate([np.zeros((1,2*T)), np.eye(1, T),np.eye(1, T)], axis=1)
                s_status_a = np.full(2,1,dtype=np.float32)
                s_status_a[1] = 0

                s_work_u = np.full(1,0,dtype=np.float32)
                s_work_m = np.concatenate([np.eye(1, T, dtype=np.float32),np.zeros((1,3*T), np.float32)], axis=1)
                s_work_a = np.full(1,0,dtype=np.float32)

            elif self.sts == 1: # 0. pillanatban elindult a motor
                s_status_u = np.full(2,1,dtype=np.float32)
                s_status_u[1] = 0
                s_status_m1 = np.concatenate([np.zeros((1,T), np.float32), np.eye(1, T, dtype=np.float32), np.zeros((1,T), np.float32), np.eye(1, T, dtype=np.float32)], axis=1)
                s_status_m2 = np.concatenate([np.zeros((1,2*T)), np.eye(1,T), np.zeros((1,T))], axis=1)
                s_status_a = np.full(2,1,dtype=np.float32)
                s_status_a[1] = 0

                s_work_u = np.full(1,-self.g_plus_max,dtype=np.float32)
                s_work_m = np.concatenate([np.zeros((1,3*T), np.float32), np.eye(1, T, dtype=np.float32)], axis=1) * -(self.g_min + self.g_plus_max)
                s_work_m[0,0] = 1
                s_work_a = np.full(1,0,dtype=np.float32)
            elif self.sts == 2: # 0. pillanatban megy a motor
                s_status_u = np.full(2,1,dtype=np.float32)
                s_status_u[1] = 0
                s_status_m1 = np.concatenate([np.zeros((1,T), np.float32), np.zeros((1,T), np.float32), np.eye(1, T, dtype=np.float32), np.eye(1, T, dtype=np.float32)], axis=1)
                s_status_m2 = np.concatenate([np.zeros((1,T)), np.eye(1,T), np.zeros((1,2*T))], axis=1)
                s_status_a = np.full(2,1,dtype=np.float32)
                s_status_a[1] = 0

                s_work_u = np.full(2,0,dtype=np.float32)
                s_work_u[1] = -np.inf
                s_work_m = np.concatenate([np.full((2,1),1),np.zeros((2,3*T-1), np.float32),np.array([[self.g_minus_max-self.g0],[-self.g_plus_max-self.g0]],np.float32), np.zeros((2,T-1))], axis=1)
                s_work_a = np.full(2,0,dtype=np.float32)
                s_work_a[0] = np.inf
            else: # 0. pillanatban leáll a motor
                s_status_u = np.full(2,0,dtype=np.float32)
                s_status_m1 = np.concatenate([np.zeros((1,T), np.float32), np.eye(1, T, dtype=np.float32), np.eye(1, T, dtype=np.float32), np.zeros((1,T), np.float32)], axis=1)
                s_status_m2 = np.concatenate([np.zeros((1,3*T)), np.eye(1,T)], axis=1)
                s_status_a = np.full(2,1,dtype=np.float32)
                s_status_a[1] = 0

                s_work_u = np.full(1,0,dtype=np.float32)
                s_work_m = np.concatenate([np.eye(1, T, dtype=np.float32),np.zeros((1,3*T), np.float32)], axis=1)
                s_work_a = np.full(1,0,dtype=np.float32)
            s_status_m = np.concatenate([s_status_m1,s_status_m2], axis=0)

            # A minimális futási és leállási időre vonatkozó kényszerek megvalósítása
            min_run_u,min_run_m,min_run_a = MILPConstraints.generate_turbine_min_run_constraints(T,self.min_on,self.et,self.min_on-self.min0_on)
            min_stop_u,min_stop_m,min_stop_a = MILPConstraints.generate_stop_time_constraints(T,self.et,self.min_off,self.min_off-self.min0_off)

            # Az indulási és a leállási időre vonatkozó kényszerek megvalósítása
            start_time_u,start_time_m,start_time_a = MILPConstraints.generate_start_time_constraints(T,self.st,self.st-self.st0)
            stop_time_u,stop_time_m,stop_time_a = MILPConstraints.generate_stop_time_constraints(T,self.et,delay=self.et-self.et0)

            u = [limit_u_work,               limit_u,control_u1,control_u2,status_u,s_status_u,min_run_u,min_stop_u,start_time_u,stop_time_u,s_work_u,sum_status_u]
            m = [limit_m1_work,limit_m2_work,limit_m,control_m1,control_m2,status_m,s_status_m,min_run_m,min_stop_m,start_time_m,stop_time_m,s_work_m,sum_status_m]
            a = [limit_a_work,               limit_a,control_a1,control_a2,status_a,s_status_a,min_run_a,min_stop_a,start_time_a,stop_time_a,s_work_a,sum_status_a]   
            under,matrix,above = MatrixMerger.Merge_MILP_Constraints(u,m,a)

            if self.st0 < self.st: # Ha a turbina a t = 0 időpillnatban még az indulási fázisban van, akkor megvalósítjuk az idevonatkozó kényszert
                intval = self.st - self.st0
                s_u,s_m,s_a = MILPConstraints.generate_constraints_to_start_time_check(T,intval,intval,intval,True)
                under,matrix,above = MatrixMerger.Merge_MILP_Constraints([under,s_u],[matrix,s_m],[above,s_a])

            if self.et0 < self.et: # Ha a turbina a t = 0 időpillnatban még a leállási fázisban van, akkor megvalósítjuk az idevonatkozó kényszert
                intval = self.et - self.et0
                e_u,e_m,e_a = MILPConstraints.generate_constraints_to_start_time_check(T,intval,intval,intval,True,True)
                under,matrix,above = MatrixMerger.Merge_MILP_Constraints([under,e_u],[matrix,e_m],[above,e_a])
        else:
            control_m1 = np.concatenate([control_m1[:,:T],np.eye(T-1,T,1)*(-self.g_min),np.zeros((T-1,T),dtype=np.float32),np.eye(T-1,T,1)*(-self.g_plus_max)],axis=1)
            # Az állapotváltozásokra vonatkozó kényszerek megvalósítása
            status_u,status_m,status_a = MILPConstraints.generate_constraints_to_status_changes(T,0.,0.)
            # A kezdeti állapotváltozásokra vonatkozó kényszerek megvalósítása
            start_status_u = np.full(1,-self.go0,dtype=np.float32)
            start_status_a = np.full(1,-self.go0,dtype=np.float32)
            start_status_m = np.zeros((1,4*T),dtype=np.float32)
            start_status_m[0][T] = 1
            start_status_m[0][2*T::T] = -1

            # A minimális futási és leállási időre vonatkozó kényszerek megvalósítása
            min_run_u,min_run_m,min_run_a = MILPConstraints.generate_sum_constraints(T,self.min_on,-1,0,delay=self.min_on-self.min0_on)
            min_stop_u,min_stop_m,min_stop_a = MILPConstraints.generate_sum_constraints(T,self.min_off,isEnd=True,delay=self.min_off - self.min0_off)

            # A létrehozott mátrixokból egy mátrixot, míg az alsó és felső becsléseket tartalmazó vektorokból egy-egy vektort készítünk
            u = [limit_u_work,               limit_u,control_u1,control_u2,status_u,start_status_u,min_run_u,min_stop_u]
            m = [limit_m1_work,limit_m2_work,limit_m,control_m1,control_m2,status_m,start_status_m,min_run_m,min_stop_m]
            a = [limit_a_work,               limit_a,control_a1,control_a2,status_a,start_status_a,min_run_a,min_stop_a]
            under,matrix,above = MatrixMerger.Merge_MILP_Constraints(u,m,a)

        if self.min0_off < self.min_off: # Ha a gázmotor/turbina a t = 0 időpillanatban még nem pihent eleget, akkor megvalósítjuk az ide vonatkozó kényszert
            intval = self.min_off-self.min0_off
            off_u,off_m,off_a = MILPConstraints.generate_constraints_to_start_time_check(T,intval)
            if self.st > 0:
                off_m[0,T:T + intval] = 1
            under,matrix,above = MatrixMerger.Merge_MILP_Constraints([under,off_u],[matrix,off_m],[above,off_a])

        if self.min0_on < self.min_on: # Ha a gázmotor/turbina a t = 0 időpillanatban még nem futott eleget, akkor megvalósítjuk az ide vonatkozó kényszert
            on_u,on_m,on_a = MILPConstraints.generate_constraints_to_start_time_check(T,self.min_on-self.min0_on,self.min_on-self.min0_on,self.min_on-self.min0_on)
            under,matrix,above = MatrixMerger.Merge_MILP_Constraints([under,on_u],[matrix,on_m],[above,on_a])

        if self.st == 0 and self.g0 != None: # Ha van kezdőérték gázotoroknál, megvalósítjuk a rávonatkozó kényszert
            start_u = np.array([-np.inf, self.g0-self.g_minus_max], dtype=np.float32)
            start_a = np.array([self.g0+self.g_plus_max, np.inf], dtype=np.float32)
            start_m = np.zeros((2,4*T),dtype=np.float32)
            start_m[0,0] = 1
            start_m[0,T] = -self.g_min
            start_m[1,2*T] = self.g_min
            start_m[1,0] = 1
            under,matrix,above = MatrixMerger.Merge_MILP_Constraints([under,start_u],[matrix,start_m],[above,start_a])

        return under,matrix,above

        

    def milp_get_cost(self, T: int):
        cf = np.concatenate([np.full(T, self.cost), np.full(T, self.startcost), np.full(T, self.endcost), np.zeros(T)])
        itr = np.concatenate([np.zeros(T), np.full(3*T, 1)])
        return cf, itr

    def update_to_milp(self):
        # Amennyiben egy gázmotor nem MILP-es paraméterekkel van megadva, de egy másik gázmotor igen, 
        # akkor ennek segítségével a nem MILP-es gázmotornak is lesznek már MILP-es adattagjai
        self.g_min = 0
        self.startcost = 0
        self.endcost = 0
        self.go0 = 1
        self.min_on = 3
        self.min_off = 3
        self.min0_on = self.min_on
        self.min0_off = self.min_off
        self.st = 0
        self.et = 0
        self.st0 = 0
        self.et0 = 0

class VPPEnergyStorage(VPPItem):
    def __init__(self,storage_min:float, storage_max:float, charge_max:float,
                 discharge_max:float, charge_loss:float, discharge_loss:float, charge_cost:float, discharge_cost:float, s0:float|None = None):
        
        # Ellenőrzések:
        if storage_min < 0. or storage_max < 0. or charge_max < 0. or discharge_max < 0.:
            raise Exception("A storage_min, storage_max, charge_max, discharge_max értékek nem lehetnek 0-nál kisebb értékek.")
        elif storage_max < storage_min:
            raise Exception("A storage_min értéke nem lehet nagyobb mint a storage_max-é!")
        elif discharge_loss < 0. or discharge_loss >= 1. or charge_loss < 0. or charge_loss >= 1.:
            raise Exception("A discharge_loss és charge_loss értékeknek legalább 0-nak és 1-nél kevesebbnek kell lennie!")

        self.storage_min = storage_min
        self.storage_max = storage_max
        self.charge_max = charge_max
        self.discharge_max = discharge_max
        self.charge_loss = charge_loss
        self.discharge_loss = discharge_loss
        self.charge_cost = charge_cost
        self.discharge_cost = discharge_cost

        # A kezdőérték beállítása
        if s0 == None:
            s0 = storage_min
        else:
            if s0 < storage_min:
                s0 = storage_min
            elif s0 > storage_max:
                s0 = self.storage_max
        self.s0 = s0

        self.is_MILP = False

    '''
    A kényszerek vektora:

    s(0)
    s(1)
    ...
    c(0)
    c(1)
    ...
    d(0)
    d(1)
    ...

    '''
    def lp_get_constraints(self, T:int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sm, sv = LinearConstraints.generate_constraints_min_max(T, self.storage_min,self.storage_max)
        cm, cv = LinearConstraints.generate_constraints_min_max(T, 0, self.charge_max)
        dm, dv = LinearConstraints.generate_constraints_min_max(T, 0, self.discharge_max)
        m, v = MatrixMerger.MergeDiff_LP_Constraints([sm, cm, dm], [sv, cv, dv])

        # Kezdő érték beállítása, ha van
        eqm = np.zeros((1, 3 * T), dtype=np.float32)
        eqm[0, 0] = 1
        eqm[0, T] = -1. + self.charge_loss
        eqm[0, 2*T] = 1. + self.discharge_loss
        eqv = np.array([self.s0], dtype=np.float32)

        # A kontinuitás felállítása
        contm, contv = LinearConstraints.generate_constraints_continuity(T, self.charge_loss, self.discharge_loss)
        eqm = np.concatenate([eqm, contm], axis=0)
        eqv = np.concatenate([eqv, contv])
        return m, v, eqm, eqv

    def lp_get_cost(self, T: int) -> np.ndarray:
        return np.concatenate([np.zeros(T, dtype=np.float32), np.full(T, (1-self.charge_loss)*self.charge_cost, dtype=np.float32), np.full(T,(1+self.discharge_loss)*self.discharge_cost, dtype=np.float32)], axis=0)

    def milp_get_constraints(self, T: int): # Elkészíti a MILP-hez szükséges kényszereket (jelölések: u/a = alsó/felső korlátokat tartalmazó vektor; m = az együtthatókat tartalmazó mátrix)
        # A korlátosságra vonatkozó kényszerek megvalósítása
        storage_u,storage_m,storage_a = MILPConstraints.generate_constraints_for_limitation(T,self.storage_max,self.storage_min)
        charge_u,charge_m,charge_a = MILPConstraints.generate_constraints_for_limitation(T,self.charge_max)
        discharge_u,discharge_m,discharge_a = MILPConstraints.generate_constraints_for_limitation(T,self.discharge_max)
        scd_u,scd_m,scd_a = MatrixMerger.MergeDiff_MILP_Constraints([storage_u,charge_u,discharge_u],[storage_m,charge_m,discharge_m],[storage_a,charge_a,discharge_a])
        
        # A kontinuitásra vonatkozó kényszerek megvalósítása
        cont_m,cont_ua = LinearConstraints.generate_constraints_continuity(T,self.charge_loss,self.discharge_loss)

        # A kezdőértékre vonatkozó kényszer megvalósítása
        start_m = np.zeros((1, 3 * T), dtype=np.float32)
        start_m[0, 0] = 1
        start_m[0, T] = -1. + self.charge_loss
        start_m[0, 2*T] = 1. + self.discharge_loss
        start_ua = np.array([self.s0], dtype=np.float32)

        # A létrehozott mátrixokból egy mátrixot, míg az alsó és felső becsléseket tartalmazó vektorokból egy-egy vektort készítünk
        u = [scd_u,cont_ua,start_ua]
        m = [scd_m,cont_m,start_m]
        a = [scd_a,cont_ua,start_ua]
        under,matrix,above = MatrixMerger.Merge_MILP_Constraints(u,m,a)
        return under,matrix,above

    def milp_get_cost(self, T: int):
        cf = self.lp_get_cost(T)
        itr = np.zeros(3*T)
        return cf, itr

class VPPSolarPanel(VPPItem):
    def __init__(self, r_max:float, delta_r_plus_max:float, delta_r_minus_max:float, cost:float, T:int, r0:float|None = None, starting:int=0, exp_v:float=13, range:float=8, value_at_end:float=0.001, addNoise:bool=True, seed=None):
        # Ellenőrzések:
        if delta_r_plus_max < 0. or r_max < 0. or delta_r_minus_max < 0. or T < 0:
            raise Exception("A delta_r_plus_max, r_max, delta_r_minus_max, T értékek nem lehetnek 0-nál kisebbek!")


        self.delta_r_plus_max = delta_r_plus_max
        self.r_max = r_max
        self.delta_r_minus_max = delta_r_minus_max
        self.cost = cost
        self.T = T
        self.exp_v = exp_v
        self.range = range
        self.addNoise = addNoise

        x = np.arange(0, T)
        deviation = math.sqrt((-0.5 * (range**2) / math.log(value_at_end)))
        self.deviation = deviation
        y = np.e**(-((x-exp_v)**2)/(2*deviation**2))
        self.original = np.copy(y)
        y = (y - value_at_end) / (np.max(y) - value_at_end)
        y[y < 0] = 0
        
        # A kezdő intervallum beállítása, ha helytelenül lenne megadva
        if starting < 0:
            starting = 0
        elif starting >= T:
            starting = T - 1

        if starting != 0:
            y = np.roll(y, -starting)

        self.r_max_values = y * r_max


        if addNoise:
            if seed != None:
                gen = np.random.default_rng(seed)
            else:
                gen = np.random.default_rng()
            noise = gen.uniform(-self.r_max_values, 0, size=T)
            self.r_max_values += noise
        if r0 != None:
            if r0 < 0:
                r0 = 0
            self.r0 = r0
        else:
            self.r0 = None
        self.is_MILP = False
    
    def lp_get_constraints(self, T:int) -> tuple[np.ndarray, np.ndarray, np.ndarray|None, np.ndarray|None]: 
        m, v = LinearConstraints.generate_constraints_min_max(T, 0, 10)
        v[:T] = self.r_max_values
        
        mmm, mmv = LinearConstraints.generate_constraints_min_max_time(T, self.delta_r_minus_max, self.delta_r_plus_max)


        # Kezdő érték beállítása, ha van
        
        if self.r0 == None:
            m, v = MatrixMerger.Merge_LP_Constraints([m, mmm], [v, mmv])
        else:
            r0m = np.zeros((2, T), dtype=np.float32)
            r0m[0, 0] = -1
            r0m[1, 0] = 1
            r0v = np.array([self.delta_r_minus_max - self.r0, self.delta_r_plus_max + self.r0], dtype=np.float32)
            m, v = MatrixMerger.Merge_LP_Constraints([m, mmm, r0m], [v, mmv, r0v])
        
        return m, v, None, None    
    def lp_get_cost(self, T:int) -> np.ndarray: 
        return np.full(T, self.cost)

    def milp_get_constraints(self, T: int): # Elkészíti a MILP-hez szükséges kényszereket (jelölések: u/a = alsó/felső korlátokat tartalmazó vektor; m = az együtthatókat tartalmazó mátrix)
        # A korlátosságra vonatkozó kényszerek megvalósítása
        limit_u,limit_m,limit_a = MILPConstraints.generate_constraints_for_limitation(T,10)
        limit_a[:T] = self.r_max_values

        # A korlátos szabályozhatóságra vonatkozó kényszerek megvalósítása
        control_u,control_m,control_a = MILPConstraints.generate_constraints_to_controllability(T,-self.delta_r_minus_max,self.delta_r_plus_max,1)
        
         # A létrehozott mátrixokból egy mátrixot, míg az alsó és felső becsléseket tartalmazó vektorokból egy-egy vektort készítünk
        under,matrix,above = MatrixMerger.Merge_MILP_Constraints([limit_u,control_u],[limit_m,control_m],[limit_a,control_a])


        if self.r0 != None: # Ha van kezdőérték, megvalósítjuk a rávonatkozó kényszert
            start_u = np.full(1,self.r0-self.delta_r_minus_max,dtype=np.float32)
            start_a = np.full(1,self.r0+self.delta_r_plus_max,dtype=np.float32)
            start_m = np.zeros((1,T),dtype=np.float32)
            start_m[0,0] = 1
            under,matrix,above = MatrixMerger.Merge_MILP_Constraints([under,start_u],[matrix,start_m],[above,start_a])

        return under,matrix,above
    
    def milp_get_cost(self, T: int):
        cf = self.lp_get_cost(T)
        itr = np.full(T, 1)
        return cf, itr


class VPPRenewable(VPPItem):
    def __init__(self, e_max:np.ndarray, cost:float, label:str):
        aclist = ["SP", "WP"] # solar power, wind power
        if label not in aclist:
            raise Exception("A lable paraméternek csak a következő értékeket veheti fel: ", aclist)

        self.e_max = e_max
        self.cost = cost
        self.label = label
        self.is_MILP = False

    def lp_get_constraints(self, T:int) -> tuple[np.ndarray, np.ndarray, np.ndarray|None, np.ndarray|None]:
        m, v = MatrixMerger.Merge_LP_Constraints([np.eye(T, T) * -1, np.eye(T, T)],[np.zeros(T), self.e_max])
        return m, v, None, None
    
    def lp_get_cost(self, T:int) -> np.ndarray:
        return np.full(T, self.cost)
    
    def milp_get_constraints(self, T:int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.zeros(T), np.eye(T, T), self.e_max
    
    def milp_get_cost(self, T:int) -> tuple[np.ndarray, np.ndarray]:
        return np.full(T, self.cost), np.zeros(T)

class LinearConstraints:

    @staticmethod
    def generate_constraints_min_max(T:int, min:float, max:float) -> tuple[np.ndarray, np.ndarray]: #sima korlátosság
        A = np.concatenate((np.eye(T),np.diag(np.full((T,), -1))),axis=0)
        b = np.concatenate([np.full(T,max), np.full(T,-1 * min)])
        return A, b


    @staticmethod
    def generate_constraints_min_max_time(T:int, min:float, max:float) -> tuple[np.ndarray, np.ndarray]: #i,t időpillanatból kivonjuk az i,t-1 időpillanatot
        min_matrix = np.zeros((T-1, T))
        np.fill_diagonal(min_matrix[:,:-1], 1)
        np.fill_diagonal(min_matrix[:,1:], -1)

        max_matrix = np.zeros((T-1, T))
        np.fill_diagonal(max_matrix[:,:-1], -1)
        np.fill_diagonal(max_matrix[:,1:], 1)

        A = np.concatenate((min_matrix,max_matrix),axis=0)
        b = np.concatenate((np.full(T - 1, min), np.full(T - 1,max)))
        return A, b

    @staticmethod
    def generate_constraints_continuity(T:int, charge_loss:float ,discharge_loss:float) -> tuple[np.ndarray, np.ndarray]:
        contm = np.zeros((T - 1, 3 * T), dtype=np.float32).reshape(-1)
        contm[0::3*T+1] = 1
        contm[1::3*T+1] = -1
        contm[T + 1::3*T + 1] = 1. - charge_loss
        contm[2*T + 1::3*T + 1] = -1. - discharge_loss
        contm = contm.reshape((T - 1, 3 * T))
        contv = np.zeros(T-1, dtype=np.float32)
        return contm, contv

class MatrixMerger:
    # =====================================
    # LP kényszerek
    # =====================================
    '''A VPP részvevőinek kényszerfeltételik és költség függvényeik összefűzését végző függvény'''
    @staticmethod
    def MergeAllVPPItem_LP_ConstraintsAndCost(itemList:list[VPPItem], d:np.ndarray, l:np.ndarray, T:int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # A <= kényszerek mátrixa és vektora
        m = list()
        v = list()

        cf = list() #költség függvény

        # Termelési kényszer sum(gi,t) +  sum(di,t) ...
        eqm = 0
        eqv = d+l

        # Az = kényszerek mátrixa és vektora
        eqml = list()
        eqvl = list()


        # Végi megyünk a VPP részvevőkön
        for e in itemList:
            ma, ve, sm, sv = e.lp_get_constraints(T) # Elkérjük a kényszeriket
            cf.append(e.lp_get_cost(T)) # a költség függvényt

            if (ma is not None):
                m.append(ma)
                v.append(ve)
            else:
                if isinstance(e, VPPEnergyStorage):
                    m.append(np.zeros((1, 3 * T), dtype=np.float32))
                    v.append(np.zeros(1, dtype=np.float32))
                else:
                    m.append(np.zeros((1, T), dtype=np.float32))
                    v.append(np.zeros(1, dtype=np.float32))

            if sm is not None:
                eqml.append(sm)
                eqvl.append(sv)
            else:
                if isinstance(e, VPPEnergyStorage):
                    eqml.append(np.zeros((1, 3 * T), dtype=np.float32))
                    eqvl.append(np.zeros(1, dtype=np.float32))
                else:
                    eqml.append(np.zeros((1, T), dtype=np.float32))
                    eqvl.append(np.zeros(1, dtype=np.float32))


                
            # összerakjuk a "Energiaigény és összenergia" pont alatti kényszert
            # A tárolók esetén figyelni kell, hogy csak a d(j,t) és c(j,t) szerepel a képletben így a többit ki kell nullázni
            if type(eqm) == int:
                if isinstance(e, VPPGasEngine) or isinstance(e, VPPSolarPanel) or isinstance(e, VPPRenewable):
                    eqm = np.eye(T, T, dtype=np.float32)
                elif isinstance(e, VPPEnergyStorage):
                    eqm = np.concatenate([np.zeros((T,T), dtype=np.float32), np.diag(np.full((T,), -1)), np.eye(T, T, dtype=np.float32)], axis=1)
                else:
                    raise Exception("Unknown VPPItem!")
            else:
                if isinstance(e, VPPGasEngine) or isinstance(e, VPPSolarPanel) or isinstance(e, VPPRenewable):
                    eqm = np.concatenate([eqm, np.eye(T, T, dtype=np.float32)], axis=1)
                elif isinstance(e, VPPEnergyStorage):
                    eqm = np.concatenate([eqm, np.zeros((T,T), dtype=np.float32), np.diag(np.full((T,), -1)), np.eye(T, T, dtype=np.float32)], axis=1)
                else:
                    raise Exception("Unknown VPPItem!")



        resm, resv = MatrixMerger.MergeDiff_LP_Constraints(m, v) # összefűzzük a <= típusú kényszereket
        eqmp, eqvp = MatrixMerger.MergeDiff_LP_Constraints(eqml, eqvl) # összefűzzük a = típusú kényszereket
        
        # Az '=' típusú kényszerek összeszerelése
        eqm = np.concatenate([eqm, eqmp], axis=0)
        eqv = np.concatenate([eqv, eqvp], axis=0)


        # A mátrixok optimalizálása
        eqm, eqv = MatrixMerger.Optimaze_LP_MatrixesAndVectors(eqm, eqv)
        resm, resv = MatrixMerger.Optimaze_LP_MatrixesAndVectors(resm, resv)
        
        return resm, resv, np.concatenate(cf, axis=0), eqm, eqv
    
    '''
    A függvény optimalizálja a megadott kényszer mátrixot:
        - Kiveszi belőle duplikált sorokat
        - Megszünteti a csupa nulla sorokat (mind a mátrixban 0 van mind a vektorban)
    '''
    @staticmethod
    def Optimaze_LP_MatrixesAndVectors(m:np.ndarray, v:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data = np.concatenate([m, v.reshape(-1, 1)], axis=1)
        uniques = np.unique(data, axis=0)
        uniques = uniques[np.logical_not(np.all(np.logical_not(uniques != 0), axis=1))]
        m = uniques[:,:-1]
        v = uniques[:,-1].reshape(-1)
        return m, v

    @staticmethod
    def MergeDiff_LP_Constraints(m:list[np.ndarray], v:list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        resv = np.concatenate(v, axis=0)
        resm = m.pop(0)
        for e in m:
            resShape = resm.shape
            newShape = e.shape
            resm = np.concatenate([resm, np.zeros((resShape[0], newShape[1]), dtype=np.float32)], axis=1) # Az eredeti mátrix jobb oldalát feltöltjük 0-ákkal
            newElem = np.concatenate([np.zeros((newShape[0], resShape[1]), dtype=np.float32), e], axis=1) # az új mátrix bal oldalát is feltöltjük 0-ákkal
            resm = np.concatenate([resm, newElem], axis=0)# A két mátrixot össze fűzzük
        return resm, resv
    '''
    Ha egy VPP résztvevőnek több kényszerfeltétele van akkor azokat lehet a függvény segítségével egyesíteni
    A függvény az 1 koordináta alapján fűzi össze őket
    '''
    @staticmethod
    def Merge_LP_Constraints(matrix:list[np.ndarray], vector:list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        m = np.concatenate(matrix, axis=0)
        v = np.concatenate(vector, axis=0)
        return m, v
    
    # =====================================
    # MILP kényszerek
    # =====================================

    '''A VPP részvevőinek kényszerfeltételik és költség függvényeik összefűzését végző függvény'''
    @staticmethod
    def MergeAllVPPItem_MILP_ConstraintsAndCost(itemList:list[VPPItem], d:np.ndarray, l:np.ndarray, T:int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        # lb <= Ax <= ub kényszerek
        lvl = list()
        uvl = list()
        ml = list()

        # Költség függvény és integritás jelölő
        cfl = list()
        itrl = list()

        # Termelést megkövetelő kényszerek
        mpro = np.zeros((T,0))



        for item in itemList:
            if not item.is_MILP and isinstance(item,VPPGasEngine):
                item.update_to_milp()
            lv, A, uv = item.milp_get_constraints(T)
            lvl.append(lv)
            ml.append(A)
            uvl.append(uv)
            cf, itr = item.milp_get_cost(T)
            cfl.append(cf)
            itrl.append(itr)

            if isinstance(item, VPPGasEngine):
                mpro = np.concatenate([mpro, np.eye(T, T), np.zeros((T,3*T))], axis=1)
            elif isinstance(item, VPPEnergyStorage):
                mpro = np.concatenate([mpro, np.zeros((T,T), dtype=np.float32), np.diag(np.full((T,), -1)), np.eye(T, T, dtype=np.float32)], axis=1)
            elif isinstance(item, VPPSolarPanel) or isinstance(item, VPPRenewable):
                mpro = np.concatenate([mpro, np.eye(T, T)],axis=1)
            else:
                raise Exception("Unknown VPPItem!")
            

        reslv, resm, resuv = MatrixMerger.MergeDiff_MILP_Constraints(lvl, ml, uvl)
        reslv, resm, resuv = MatrixMerger.Merge_MILP_Constraints([reslv, l], [resm, mpro], [resuv, l])
        rescf = np.concatenate(cfl) # Költség fv
        resitr = np.concatenate(itrl) # Intr

        return reslv, resm, resuv, rescf, resitr
    
    '''
    A függvény optimalizálja a megadott kényszer mátrixot:
        - Kiveszi belőle duplikált sorokat
        - Megszünteti a csupa nulla sorokat (mind a mátrixban 0 van mind a vektorban)
    '''
    @staticmethod
    def Optimaze_MILP_MatrixesAndVectors(vl:np.ndarray, m:np.ndarray, vu:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data= np.concatenate([vl.reshape(-1, 1), m, vu.reshape(-1, 1)], axis=1)
        uniques = np.unique(data, axis=0)
        uniques = uniques[np.logical_not(np.all(np.logical_not(uniques != 0), axis=1))]
        m = uniques[:,1:-1]
        vu = uniques[:,-1].reshape(-1)
        vl = uniques[:, 0].reshape(-1)
        return vl, m, vu

    '''
    Különböz típusú mátrixokat merge-el (A két mátrixot elhelyezi a diagonális észelbe és a többi elemet kinullázza)
    '''
    @staticmethod
    def MergeDiff_MILP_Constraints(vl:list[np.ndarray], m:list[np.ndarray], vu:list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resvl = np.concatenate(vl, axis=0)
        resvu = np.concatenate(vu, axis=0)
        resm = m.pop(0)
        for e in m:
            resShape = resm.shape
            newShape = e.shape
            resm = np.concatenate([resm, np.zeros((resShape[0], newShape[1]), dtype=np.float32)], axis=1) # Az eredeti mátrix jobb oldalát feltöltjük 0-ákkal
            newElem = np.concatenate([np.zeros((newShape[0], resShape[1]), dtype=np.float32), e], axis=1) # az új mátrix bal oldalát is feltöltjük 0-ákkal
            resm = np.concatenate([resm, newElem], axis=0)# A két mátrixot össze fűzzük
        return resvl, resm, resvu
    '''
    Ha egy VPP résztvevőnek több kényszerfeltétele van akkor azokat lehet a függvény segítségével egyesíteni
    A függvény az 1 koordináta alapján fűzi össze őket
    '''
    @staticmethod
    def Merge_MILP_Constraints(vectorl:list[np.ndarray], matrix:list[np.ndarray], vectoru:list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = np.concatenate(matrix, axis=0)
        vl = np.concatenate(vectorl, axis=0)
        vu = np.concatenate(vectoru, axis=0)
        return vl, m, vu
    


class MILPConstraints:
    @staticmethod
    # Létrehozza az összegzéshez szükséges mátrixot
    def generate_sum_matrix(T:int,height:int,interval:int,isSub:bool = False) -> np.ndarray:
        row = np.zeros(T,dtype=np.float32)
        row[:interval] = 1
        if T-interval > 0:
            row[interval] = -interval if isSub else 0
        col = np.zeros(height,dtype=np.float32)
        col[0] = 1
        sum_m = toeplitz(col,row)
        return sum_m

    @staticmethod
    # Olyan mátrixot hoz létre, amely a szomszédos elemek különbségét tartalmazza egy adott t időpillanattól kezdve
    def generate_delta_matrix(T:int,height:int,multiply:int,isOpp:bool = False) -> np.ndarray:
        row = np.zeros(T,dtype=np.float32)
        row[T-height-1] = -multiply if isOpp else multiply
        row[T-height] = multiply if isOpp else -multiply
        col = np.zeros(height,dtype=np.float32)
        col[0] = row[0] if multiply == 1 else 0
        delta_m = toeplitz(col,row)
        return delta_m

    # A minimális futási/leállási időhöz szükséges mátrixot és vektorokat állítja elő
    @staticmethod
    def generate_sum_constraints(T:int,min_time:int,lower:int=0,upper:int=1,isEnd:bool = False,delay:int = 0) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        height = T-min_time + 1
        under = np.full(height,lower,dtype=np.float32)
        above = np.full(height,upper,dtype=np.float32)
        A_work = np.zeros((height,T))
        A_start = MILPConstraints.generate_sum_matrix(T,height,min_time)
        A_end = np.zeros((height,T))
        on_val = -1.
        if isEnd:
            tmp = A_start
            A_start = A_end
            A_end = tmp
            on_val = 1.
        A_on = np.concatenate((np.zeros((height,min_time-1)),np.diag(np.full(height,on_val))),axis=1)
        A = np.concatenate((A_work,A_start,A_end,A_on),axis=1)
        return under[delay:],A[delay:,:],above[delay:]
    
    # A korlátossághoz tartozó mátrixot és vektorokat állítja elő
    @staticmethod
    def generate_constraints_for_limitation(T:int,upper:float,lower:float=0.) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        under = np.full(T,lower)
        above = np.full(T,upper)
        A = np.eye(T,T)
        return under,A,above

    # A korlátos szabályozhatósághoz kapcsolódó mátrixot és vektorokat állítja elő
    @staticmethod
    def generate_constraints_to_controllability(T:int,lower:float,upper:float,unknown_variable:int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        under = np.full(T-1,lower)
        above = np.full(T-1,upper)
        A = MILPConstraints.generate_delta_matrix(T,T-1,1,True)
        A = np.concatenate((A,np.zeros((T-1,(unknown_variable-1)*T),dtype=np.float32)),axis=1)
        return under,A,above
    
    # Az állapotváltozásokhoz kapcsolódó mátrixot és vektorokat állítja elő
    @staticmethod
    def generate_constraints_to_status_changes(T:int,lower:float=0.,upper:float=0.) ->tuple[np.ndarray,np.ndarray,np.ndarray]:
        under = np.full(T-1,lower)
        above = np.full(T-1,upper)
        A_work = np.zeros((T-1,T))
        A_start = np.eye(T-1,T,1)
        A_end = np.concatenate((np.zeros((T-1,1),np.float32),np.diag(np.full(T-1,-1,np.float32))),axis=1)
        A_on = MILPConstraints.generate_delta_matrix(T,T-1,1)
        A = np.concatenate((A_work,A_start,A_end,A_on),axis=1)
        return under,A,above
       
    # A plusz futási, illetve pihenési időt megvalósító kényszerhez szükséges mátrixot és vektorokat állítja elő
    @staticmethod
    def generate_constraints_to_start_time_check(T:int,interval:int,lower:float=0.,upper:float=0.,isTurbine:bool = False,isTurbineEnd:bool = False) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        zero = np.zeros((1,T),dtype=np.float32)
        A = MILPConstraints.generate_sum_matrix(T,1,interval)
        if isTurbine:
            if isTurbineEnd:
                A = np.concatenate([zero,zero,A,zero],axis=1)
            else:
                A = np.concatenate([zero,A,zero,zero],axis=1)
        else:
            A = np.concatenate((np.zeros((1,3*T)),A),axis=1)
        return np.full(1,lower),A,np.full(1,upper)
    
    # A gázturbinák esetén az indulási időhöz vagy a minimális futási időhöz szükséges mátrixot és a hozzá tartozó vektorokat állítja elő
    @staticmethod
    def generate_start_time_constraints(T:int,st:int,delay:int = 0) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        height = T-st
        under_a = np.full(height,-np.inf,dtype=np.float32)
        above_a = np.zeros(height,dtype=np.float32)
        matrix_au = MILPConstraints.generate_sum_matrix(T,height,st,True)
        zero = np.zeros(matrix_au.shape,dtype=np.float32)
        e = np.eye(height,T,st,dtype=np.float32)
        e[e>0] = -st
        matrix_a = np.concatenate([zero,matrix_au,zero,e],axis=1)

        under_u = np.zeros(height,dtype=np.float32)
        above_u = np.full(height,np.inf,dtype=np.float32)
        d = MILPConstraints.generate_delta_matrix(T,height,st)
        matrix_au[matrix_au == -st] = 0
        matrix_u = np.concatenate([zero,matrix_au,zero,d],axis=1)

        under,matrix,above = MatrixMerger.Merge_MILP_Constraints([under_a,under_u],[matrix_a,matrix_u],[above_a,above_u])
        return under[delay:],matrix[delay:,:],above[delay:]
    
    # A gázturbinák esetén a leállási időhöz vagy a minimális leállási időhöz szükséges mátrixot és a hozzá tartozó vektorokat állítja elő
    @staticmethod
    def generate_stop_time_constraints(T:int,et:int,minoff:int|None = None,delay:int = 0) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        intval = T - et if minoff is None else T - minoff
        under_a = np.full(intval,-np.inf,dtype=np.float32)
        above_a = np.full(intval,et,dtype=np.float32)
        matrix_au = MILPConstraints.generate_sum_matrix(T,intval,et) if minoff is None else MILPConstraints.generate_sum_matrix(T,intval,minoff)
        e = np.eye(intval,T,et,dtype=np.float32) if minoff is None else np.eye(intval,T,minoff,dtype=np.float32)
        e[e>0] = et
        zero = np.zeros(matrix_au.shape,dtype=np.float32)
        matrix_a = np.concatenate([zero,zero,matrix_au,e],axis=1) if minoff is None else np.concatenate([zero,e,matrix_au,zero],axis=1)

        under_u = np.zeros(intval,dtype=np.float32)
        above_u = np.full(intval,np.inf,dtype=np.float32)
        d = MILPConstraints.generate_delta_matrix(T,T-et,et,True)
        matrix_u = (matrix_au + d) if minoff is None else matrix_au
        matrix_u = np.concatenate([zero,zero,matrix_u,zero],axis=1)

        under,matrix,above = MatrixMerger.Merge_MILP_Constraints([under_a,under_u],[matrix_a,matrix_u],[above_a,above_u])
        return under[delay:],matrix[delay:,:],above[delay:]

    @staticmethod
    def generate_turbine_min_run_constraints(T:int,min_on:int,et:int,delay:int = 0) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        height = T-min_on
        under_a = np.full(height,-np.inf)
        under_u = np.zeros(height)
        sum_m = MILPConstraints.generate_sum_matrix(T,T-min_on,min_on)
        above_u = np.full(height,np.inf)
        d = MILPConstraints.generate_delta_matrix(T,height,min_on)
        if min_on > et:
            above_a = np.full(height,min_on-et)
            e = np.eye(height,T,min_on)*(-et)
        else:
            above_a = np.zeros(height)
            e = np.eye(height,T,min_on)*(-min_on)
        me = sum_m + e
        matrix_a = np.concatenate([np.zeros((height,2*T)),e,me],axis=1)
        matrix_u = np.concatenate([np.zeros((height,2*T)),d,sum_m],axis=1)

        under,matrix,above = MatrixMerger.Merge_MILP_Constraints([under_a,under_u],[matrix_a,matrix_u],[above_a,above_u])
        return under[delay:],matrix[delay:],above[delay:]

def solve(items:list[VPPItem], T:int, l:np.ndarray, d:np.ndarray, debug=False, detaild_chart=False, detaild_items=[], return_res=False, tofile=None, checker=None, main_title:str="", display=True):
    import charts as c
    ismilp = False
    for i in items:
        if (i.is_MILP):
            ismilp = True
            break
    if (not ismilp):
        st = time.time()
        cm, cv, cf, sc, scv = MatrixMerger.MergeAllVPPItem_LP_ConstraintsAndCost(items, d, l, T)
        ed = time.time()
        res = linprog(cf, A_ub=cm, b_ub=cv, A_eq=sc, b_eq=scv)
        if debug:
            if checker is not None:
                checkres_leq = np.sum(cm * checker, axis=1) <= cv
                checkres_eq = np.sum(sc * checker, axis=1) <= scv
            print("Ge matrix and vector:")
            for i in range(cm.shape[0]):
                print(cm[i], sep=', ', end = '')
                if checker is not None:
                    print(" <= ", cv[i], sep='', end="")
                    print(" | ", checkres_leq[i])
                else:
                    print(" <= ", cv[i], sep='')
                
            print("\n\nEqv matrix and vector:")
            for i in range(sc.shape[0]):
                print(sc[i], sep=', ', end = '')
                if checker is not None:
                    print(" = ", scv[i], sep='', end="")
                    print(" | ", checkres_eq[i])
                else:
                    print(" = ", scv[i], sep='')
                
            print("\n\nKöltség függvény: ", cf, end="\n\n")
            print("\n\nAz optimalizáló eredménye: ", res, end="\n\n")

            if tofile is not None:
                f = open(tofile, "w")
                f.write("Ge matrix and vector:")
                for i in range(cm.shape[0]):
                    f.write(np.array2string(cm[i], separator='\t') + " <= " + np.array2string(cv[i], separator='\t'))
                f.write("\n\nEqv matrix and vector:")
                for i in range(sc.shape[0]):
                    f.write(np.array2string(sc[i], separator='\t') + " = " + np.array2string(scv[i], separator='\t'))
                f.write("\n\nKöltség függvény: " + np.array2string(cf, separator='\t') + "\n\n")
                f.close()

        if res.success:
            if display:
                if detaild_items == [] and detaild_chart:
                    detaild_items = items
                c.chart(T, items, cf, res.x, l, d, detaild_chart, detaild_items, main_title=main_title)
            if return_res:
                return True, res, cf, False, ed-st
            return True
        else:
            if display:
                print("Az optimalizáció nem végrehajtható!\nHibaüzenet:\n", res.message)
            if return_res:
                return False, res, 0, False, ed-st
            return False
    else:
        st = time.time()
        lb, m, ub, cf, itr = MatrixMerger.MergeAllVPPItem_MILP_ConstraintsAndCost(items, d, l, T)
        res = milp(c=cf, integrality=itr, constraints=LinearConstraint(m, lb, ub))
        ed = time.time()
        if debug:
            if checker is not None:
                checkres = np.logical_and((lb <= np.sum(m * checker, axis=1)), (np.sum(m * checker, axis=1) <= ub))
            print("Kényszerek:")
            for i in range(m.shape[0]):
                print(lb[i], " <= ", sep='')
                print(m[i], sep=', ', end = '')
                print(" <= ", ub[i], sep='')
                if checker is not None:
                    print(checkres[i])
            print("Integritás (0-valós, 1-egész): ", itr, sep="\n")
            print("\n\nKöltség függvény: ", cf, end="\n\n")
            print("\n\nAz optimalizáló eredménye: ", res, end="\n\n")
            if tofile is not None:
                f = open(tofile, "w")
                f.write("Kényszerek:\n")
                if checker is not None:
                    matrix = np.concatenate([lb.reshape(m.shape[0], 1), np.full((m.shape[0], 1), "<="),np.char.mod('%f', m),np.full((m.shape[0], 1), "<="), ub.reshape(m.shape[0], 1), np.full((m.shape[0], 1), "|"), checkres.reshape(checkres.shape[0], 1)], axis=1)
                else:
                    matrix = np.concatenate([lb.reshape(m.shape[0], 1), np.full((m.shape[0], 1), "<="),np.char.mod('%f', m),np.full((m.shape[0], 1), "<="), ub.reshape(m.shape[0], 1)], axis=1)
                f.write(np.array2string(matrix, separator="\t").replace("[", "").replace( "]", "").replace("'",""). replace(".000000","."))
                f.write("\nIntegritás (0-valós, 1-egész):\n" + np.array2string(itr, separator='\t'))
                f.write("\n\nKöltség függvény:\n" + np.array2string(cf, separator='\t'))
                f.close()
           

        if res.success:
            if display:
                if detaild_items == [] and detaild_chart:
                    detaild_items = items
                c.chart(T, items, cf, res.x, l, d, detaild_chart, detaild_items, True, main_title)
            if return_res:
                return True, res, cf, True, ed-st
            return True
        else:
            if display:
                print("Az optimalizáció nem végrehajtható!\nHibaüzenet:\n", res.message)
            if return_res:
                return False, res, 0, True, ed-st
            return False
        
def gen_random_demand(T:int, min:int, max:int, seed:int|None=None):
    if seed != None:
        gen = np.random.default_rng(seed)
    else:
        gen = np.random.default_rng()
    return gen.integers(min, max, (T))