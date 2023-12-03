import numpy as np
import itertools as it

# Meghatározza a Mean Squared Error-ok minimumát, valamint visszadja,
# hogy ilyenkor mekkorák az eltérések az összehasonlított egységek között,
# továbbá megkapjuk a permutált egységek sorrendjét is
def min_of_mse(items1:np.ndarray[np.ndarray],items2:np.ndarray[np.ndarray]):
   if items1.size == 0 or items2.size == 0:
      return 0,np.array([[]], ndmin=2),np.zeros(0)
   if items1.shape[0] >= 9 or items2.shape[0] >= 9:
      return np.inf,np.array([[]], ndmin=2),np.zeros(0)
   permutation = np.array(list(it.permutations(items2)))
   indexes = np.array(list(it.permutations(range(items2.shape[0]))))
   differences = items1 - permutation
   diff_square = np.power(differences,2)
   means = np.mean(diff_square.reshape(permutation.shape[0],-1),axis = 1)
   arg = np.argmin(means)
   return np.min(means),differences[arg],indexes[arg]


# Meghatározza a Mean Absolute Error-ok minimumát, valamint visszadja,
# hogy ilyenkor mekkorák az eltérések az összehasonlított egységek között,
# továbbá megkapjuk a permutált egységek sorrendjét is
def min_of_mae(items1:np.ndarray[np.ndarray],items2:np.ndarray[np.ndarray]):
   if items1.size == 0 or items2.size == 0:
      return 0,np.zeros(0),np.zeros(0)
   if items1.shape[0] >= 9 or items2.shape[0] >= 9:
      return np.inf,np.zeros(0),np.zeros(0)
   permutation = np.array(list(it.permutations(items2)))
   indexes = np.array(list(it.permutations(range(items2.shape[0]))))
   differences = items1 - permutation
   diff_abs = np.abs(differences)
   means = np.mean(diff_abs.reshape(permutation.shape[0],-1),axis = 1)
   arg = np.argmin(means)
   return np.min(means),differences[arg],indexes[arg]


# Egy közelítést ad a Mean Squared Error-ok minimumára, valamint visszaadja,
# hogy az egyes egységek között mekkorák az eltérések,
# illetve a megfelelő párosítások indexeit is megkapjuk
def approx_min_of_mse(items1:np.ndarray[np.ndarray],items2:np.ndarray[np.ndarray]):
   possible = np.arange(items2.shape[0])
   result_values = np.zeros(items2.shape[0])
   diff_values = np.zeros(items2.shape[0])
   result_index = np.zeros(items2.shape[0])
   for i in range(items1.shape[0]):
      actual = items1[i]
      diff = items2 - actual
      diff_square = np.power(diff,2)
      sum_square = np.sum(diff_square,axis=1)
      min_index = possible[0]
      for j in range(possible.shape[0]):
         if sum_square[min_index] > sum_square[j]:
            min_index = j
      np.delete(possible,min_index)
      result_values[i] = sum_square[min_index]
      diff_values[i] = diff[min_index]
      result_index[i] = min_index

   return np.mean(result_values),diff_values,result_index



# Egy közelítést ad a Mean Absolute Error-ok minimumára, valamint visszaadja,
# hogy az egyes egységek között mekkorák az eltérések,
# illetve a megfelelő párosítások indexeit is megkapjuk
def approx_min_of_mae(items1:np.ndarray[np.ndarray],items2:np.ndarray[np.ndarray]):
   possible = np.arange(items2.shape[0])
   result_values = np.zeros(items2.shape[0])
   diff_values = np.zeros(items2.shape[0])
   result_index = np.zeros(items2.shape[0])
   for i in range(items1.shape[0]):
      actual = items1[i]
      diff = items2 - actual
      diff_abs = np.abs(diff)
      sum_abs = np.sum(diff_abs,axis=1)
      min_index = possible[0]
      for j in range(possible.shape[0]):
         if sum_abs[min_index] > sum_abs[j]:
            min_index = j
      np.delete(possible,min_index)
      result_values[i] = sum_abs[min_index]
      diff_values[i] = diff[min_index]
      result_index[i] = min_index

   return np.mean(result_values),diff_values,result_index

# Megadja, hogy ha minden egységet minden egységgel összehasonlíthatnánk, függetlenül attól, 
# hogy valamely egységgel már korábban összehasonlítottuk, akkor mennyi lenne a Mean Squared Error minimuma.
# Ezen felül megadja a megfelelő párosítások különbségeit, illetve a párosítások indexeit is.

def lower_limit_of_mse(items1:np.ndarray[np.ndarray],items2:np.ndarray[np.ndarray]):
   result_values = np.zeros(items2.shape[0])
   diff_values = np.zeros(items2.shape[0])
   result_index = np.zeros(items2.shape[0])
   for i in range(items1.shape[0]):
      actual = items1[i]
      diff = items2 - actual
      diff_square = np.power(diff,2)
      sum_square = np.sum(diff_square,axis=1)
      min_index = 0
      for j in range(items2.shape[0]):
         if sum_square[min_index] > sum_square[j]:
            min_index = j
      result_values[i] = sum_square[min_index]
      diff_values[i] = diff[min_index]
      result_index[i] = min_index

   return np.mean(result_values),diff_values,result_index
   
# Megadja, hogy ha minden egységet minden egységgel összehasonlíthatnánk, függetlenül attól, 
# hogy valamely egységgel már korábban összehasonlítottuk, akkor mennyi lenne a Mean Absolute Error minimuma.
# Ezen felül megadja a megfelelő párosítások különbségeit, illetve a párosítások indexeit is.

def lower_limit_of_mae(items1:np.ndarray[np.ndarray],items2:np.ndarray[np.ndarray]):
   result_values = np.zeros(items2.shape[0])
   diff_values = np.zeros(items2.shape[0])
   result_index = np.zeros(items2.shape[0])
   for i in range(items1.shape[0]):
      actual = items1[i]
      diff = items2 - actual
      diff_abs = np.abs(diff)
      sum_abs = np.sum(diff_abs,axis=1)
      min_index = 0
      for j in range(items2.shape[0]):
         if sum_abs[min_index] > sum_abs[j]:
            min_index = j
      result_values[i] = sum_abs[min_index]
      diff_values[i] = diff[min_index]
      result_index[i] = min_index

   return np.mean(result_values),diff_values,result_index