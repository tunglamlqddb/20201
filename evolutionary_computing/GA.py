import numpy as np
import math, time
import random
from collections import defaultdict

random.seed(11)
duplicate = []
i = 0
while i < 500:
    x = random.randint(0,1000)
    y = random.randint(0,1000)
    if (x,y) not in duplicate and (x,y) != (500,500):
        duplicate.append((x,y))
        i += 1
duplicate.append((500,500))
# print(duplicate)

locations = duplicate
def create_distance():
  dmax = 0
  distance = [[0 for _ in range(N+1)] for _ in range(N+1)]   # include BS
  for i in range(N+1):
    for j in range(N+1):
      distance[i][j] = math.sqrt((locations[i][0]-locations[j][0])**2 + (locations[i][1]-locations[j][1])**2)
      dmax = max(dmax, distance[i][j])
  return distance, dmax

N = 500
M = 35

# Sensor node properties
E_MAX = 10800 # max energy of sensor
E_MIN = 540 # min energy of sensor
distance, D_MAX = create_distance() # max distance between 2 nodes 
p_r = 5   # received power of node   
t_s = 10000   # battery charging/replacement time of MC ==> NOT CONFIRMED, (s)
t_d = 5 * t_s   # time interval between two adjacent rounds ==> NOT CONFIRMED
# consumption_energy_rate = []
# for i in range(N):
#     random.seed(i)
#     consumption_energy_rate.append(random.uniform(0.01, 0.1))
# consumption_energy_rate = np.array(consumption_energy_rate)
# print(consumption_energy_rate)
# consumption_energy_rate = np.array(create_consumption_energy_rate(distance))

# MC properties
E = 108000
v = 5
e = 1     # moving related coefficient  
charging_efficency = 1         #==> NOT CONFIRMED
p0 = p_r / charging_efficency
max_energy_and_time = (E_MAX - E_MIN) / charging_efficency + D_MAX * e
replacing = t_s + 2 * D_MAX / v # assume that v is the same of all the MCs
charging_and_moving_time = (E_MAX - E_MIN) / p_r + D_MAX / v
PHI = int(replacing / charging_and_moving_time) + 1

def find_sigma_in_cycle(E: np.array, n: int):    # E la mang co M phan tu
  status = [1] * M   # check dead or alive of MC
  min_num_of_rounds = (E / max_energy_and_time).astype(int)
  #min_num_of_rounds = np.array([4,4,4,4,1,2])    # test thử
  res = []
  max_rounds_per_cycle = 0
  num_of_avail_MCs = 0
  while True:
    max_rounds_per_cycle += 1
    num_of_avail_MCs += sum(status)
    res.append(num_of_avail_MCs)
    if num_of_avail_MCs >= n:
      return max_rounds_per_cycle, res
    else: 
      min_num_of_rounds -= 1
      for i in range(M):
        if min_num_of_rounds[i] == 0:
          status[i] = 0
        elif min_num_of_rounds[i] < 0:
          if min_num_of_rounds[i] + PHI == 0:
            status[i] = 1
            min_num_of_rounds[i] = (E[i] / max_energy_and_time).astype(int)

def list_nodes_require_charging_in_cycle_k(sigma_k):
  required_charging = [1] * N
  n_i = 0
  for i in range(N):
    for j in range(len(array_sum_avai_MCs_in_cycle_k)):
      if array_sum_avai_MCs_in_cycle_k[j] >= (N-n_i):
         sigma_i_1 = j
         break
    if L_K[i] >= (sigma_i_1 + sigma_k_next - 1) * max_energy_and_time + t_d:
      # print((sigma_i_1 + sigma_k_next - 1) * max_energy_and_time + t_d)
      required_charging[i] = 0
    else:
      n_i += 1
  return required_charging

def find_index_of_avai_MCs(E_l):      # in round L, E_l gom M phan tu
  res = []
  for i in range(M):
    if E_l[i] >= max_energy_and_time + D_MAX * e:
      res.append(i)
      round_min_num_of_rounds[i] = 608  
    else:
      if round_min_num_of_rounds[i] == 608:   # this MS needs to return to BS RIGHT AT THIS ROUND
        round_min_num_of_rounds[i] = 0
  return res

def update_after_MPSP(list_avai_MCs, E_l, consumed, E_after_charging):       # hàm này được chạy trước hàm find_index_of_avai_MCs, để đảm đã bao gồm các MC có thể quay trở lại từ BS
  E_l_plus_1 = [0] * M
  for j in list_avai_MCs:
    E_l_plus_1[j] = E_l[j] - consumed[j]

  list_not_avai_MCs = set(list(range(M))).difference(set(list_avai_MCs))
  for j in list_not_avai_MCs:
    round_min_num_of_rounds[j] -= 1
    if round_min_num_of_rounds[j] + PHI == 0:
      E_l_plus_1[j] = E_after_charging           # cập nhật những MC có thể quay trở lại từ BS
    else:
      E_l_plus_1[j] = E_l[j]
  
  return E_l_plus_1

round_min_num_of_rounds = [608] * M
def find_sigma_i_k_in_round(E_l: np.array, i, E_after_charging):   # cần cập nhật E_l gồm những thằng MC trở lại; E_l gom M phan tu; i là index node đang xét trong RCS
  list_avai_MCs = find_index_of_avai_MCs(E_l)    # => E_l gom M phan tu
  status = [0] * M
  temp = np.zeros(M)
  for j in range(M):
    if j in list_avai_MCs:
      temp[j] = E_l[j] / max_energy_and_time
      status[j] = 1 
    else:
      temp[j] = round_min_num_of_rounds[j]
  left_rounds_in_cycle = 0
  num_of_avail_MCs = 0
  res = []
  while True:
    left_rounds_in_cycle += 1
    num_of_avail_MCs += sum(status)
    res.append(num_of_avail_MCs)
    if num_of_avail_MCs >= n_refine - (i + 1) + 1:    # i chưa phải là index trong refine sequence ==> tìm index của i trong đó
      return left_rounds_in_cycle, res
    else:
      temp -= 1  
      for j in range(M):
        if temp[j] == 0:
          temp[j] = 0
          status[j] = 0
        elif temp[j] < 0:
          if temp[j] + PHI == 0:
            status[j] = 1 
            temp[j] = E_after_charging

def calculate_E_i_low_and_E_i_high(E_l, i, sigma_k_next, e_i, total_time_before, E_after_charging):     # of each node; E_l gom M phan tu; i là index trong RCS
  total_energy_comsumed_before = e_i - total_time_before * consumption_energy_rate[refined_charging_sequence[i]]    # chú ý consumption_energy_rate cũng phải sắp xếp theo thứ tự của L_K
  
  # E_i_low
  max_waiting_time = (find_sigma_i_k_in_round(E_l, i, E_after_charging)[0] + sigma_k_next - 1) * max_energy_and_time + t_d
  max_energy_consumed_while_waiting_for_next_charge = max_waiting_time * consumption_energy_rate[refined_charging_sequence[i]]
  E_i_low = max_energy_consumed_while_waiting_for_next_charge - total_energy_comsumed_before
  # print('sigma_i_k: {}'.format(find_sigma_i_k_in_round(E_l, i, E_after_charging)[0]))
  # print('-----------')
  # E_i_high
  E_i_high = E_MAX - total_energy_comsumed_before

  # print("*********")
  # print("DEBUG: {} - ({} - {}*{})".format(E_MAX, e_i, total_time_before, consumption_energy_rate[refined_charging_sequence[i]]))
  # print("*********")

  return E_i_low, E_i_high

def update_refine(E_l, e_init_of_node, total_time_before, index_last_node_charged_so_far, E_after_charging):     
  h_l_k = len(find_index_of_avai_MCs(E_l))
  index_last_node_charged = index_last_node_charged_so_far
  i = index_last_node_charged_so_far + 1  # tiếp tục cập nhật
  cnt = 0
  while i < n_refine:
    if calculate_E_i_low_and_E_i_high(E_l, i, sigma_k_next, e_init_of_node[refined_charging_sequence[i]], total_time_before, E_after_charging)[0] < 0:
      charged[i] = 0
    else:
      index_last_node_charged = i
      cnt += 1
    if cnt == min(h_l_k, n_refine - index_last_node_charged_so_far - 1): 
      break
    i += 1
  n_charged_before = 0  
  for i in range(index_last_node_charged_so_far + 1):
    if charged[i] == 1:
      n_charged_before += 1
  n_uncharged_before = index_last_node_charged_so_far - n_charged_before + 1
  return index_last_node_charged, n_charged_before, n_uncharged_before

def calculate_time_threshold(E_l, total_time_before, l, n_charged_before, n_uncharged_before, E_after_charging, n_refine):  # theta_k_l: round l of cycle k
  threshold = 99999999
  h_l_k = len(find_index_of_avai_MCs(E_l))
  index_of_node_in_round_L = n_charged_before + n_uncharged_before + h_l_k-1 # +1 to +h_l_k
  tests = []
  for idx in range(h_l_k):
      tests.append(n_charged_before + n_uncharged_before + idx)  
  left_rounds = []
  for test in tests:
    left_rounds.append(find_sigma_i_k_in_round(E_l, test, E_after_charging)[0])
  num_of_residual_rounds, sum_m_arr = find_sigma_i_k_in_round(E_l, index_of_node_in_round_L, E_after_charging) # include round L
  #num_of_residual_rounds -= 1   # exclude round L
  # print("*********")
  # print("DEBUG")
  # print ("Số round còn lại: {}".format(num_of_residual_rounds))
  # print("Kiểm tra số round còn lại với các node trong round hiện tại: 1 tới h_l_k: ")
  # for left_round in left_rounds:
  #   print(left_round,end=' ')
  # print("\n*********")
  # print("DEBUG: THRESHOLD: num_of_residual_rounds: {}".format(num_of_residual_rounds))
  for j in range(l+1, l + num_of_residual_rounds + 1):
    # print("DEBUG: THRESHOLD: j = {}".format(j))
    v_j = n_charged_before + n_uncharged_before + min(n_refine-n_charged_before-n_uncharged_before-1, sum_m_arr[j-l-1]) #+ 1
    # print("DEBUG time threshold: v_j = {} + {} + {}".format(n_charged_before, n_uncharged_before, sum_m_arr[j-l-1]))
    w_j = L_K[v_j] - total_time_before - (j - l) * max_energy_and_time - D_MAX / v
    # print("DEBUG time threshold: v_j = {}, w_j = {} - {} - {}*{}- {} ".format(v_j, L_K[v_j], total_time_before, (j-l), max_energy_and_time, D_MAX/v))
    if (w_j < 0):
      print("HERE")
    threshold = min(threshold, w_j) 
  return min(threshold, max_energy_and_time)

def calculate_total_time_for_round_L(q, t, g, h_k_l, p_k_l):  
  res = 0
  for i in range(p_k_l):
    for j in range(h_k_l):
      if q[i][j].solution_value() > 0:
       # print("Node {}, time {} + {} ".format(new_index_node[i], t[i][j].solution_value(), g[i][j].solution_value()))
        res = max(res, t[i][j].solution_value() + g[i][j].solution_value())
  return res

def get_position_of_MCs_in_round_L(q,t,g,h_k_l,p_k_l, new_index_MC, new_index_node):
  position = defaultdict(int)
  for j in range(h_k_l):
    for i in range(p_k_l):
      if q[i][j].solution_value() > 0:
        position[new_index_MC[j]] = new_index_node[i]
  return position

# Try solving the first round
from omc_ga import *

plot_e_k = []
total_energy = 0
round_energy = []


# Bắt đầu mỗi cycle, tìm xem tới thơi điểm nào sẽ có node cần sạc: tức năng lượng ban đầu của mỗi node trong cycle đầu tiên
np.random.seed(61)
e_k = np.random.uniform(low=2000, high=E_MAX, size=(N,))           # random nang luong khoi tao cua moi node o cycle dau
consumption_energy_rate = []
for i in range(N):
    random.seed(i+1)
    consumption_energy_rate.append(random.uniform(0.01, 0.1))
consumption_energy_rate = np.array(consumption_energy_rate)
L_K = (e_k - E_MIN) / consumption_energy_rate                      # lifetime của các node chưa xếp theo thứ tự RCS
# sắp xếp các node theo thứ tự tăng dần thời gian sống
data = [(l,idx) for (idx,l) in enumerate(L_K)]
data.sort(key=lambda tup: tup[0])  # sorts in place
tmp = np.array(data)[:, 1].astype(int)
consumption_energy_rate = consumption_energy_rate[tmp]
e_k = e_k[tmp]
L_K = np.array(data)[:, 0]
dynamic_e_k = np.copy(e_k)   # to trace e_k after every round, used for t_l_max

# print("Năng lượng tối đa dành cho di chuyển và sạc của 1 MC: {}".format(max_energy_and_time))
# print("------------")
# print("Năng lượng khởi tạo của mỗi Node (TẤT CẢ) ĐÃ sắp xếp thứ tự theo lifetime, cùng consumption rate:")
# for index, value in enumerate(e_k):
#   print("\tNode {}, energy {} -- rate {}".format(index, value, consumption_energy_rate[index]))
# print("Lifetime của mỗi Node (TẤT CẢ) theo thứ tự tăng dần:")
# for index, value in enumerate(L_K):
#   print("\tNode {}, life time {}".format(index, value))

# print("------------") 
# print("Bắt đầu giải")
# print("------------")

sigma_k, array_sum_avai_MCs_in_cycle_k = find_sigma_in_cycle(np.array([E]*M), N)
sigma_k_next = sigma_k  # giả sử năng lượng khởi tạo ở cycle k và k + 1 là bằng nhau
# update Refined Charging Sequence
is_charged = list_nodes_require_charging_in_cycle_k(sigma_k)    # len N
refined_charging_sequence = [i for i in range(N) if is_charged[i] == 1] # INDICES of nodes require charging due to lemma 2.2; original index in L_K
# suy ra
n_refine = len(refined_charging_sequence)
round_index = [0] * n_refine  # Lưu lại round mà node sẽ được sạc: nhưng đây chỉ là kết quả dự đoán đầu mỗi cycle, được cập nhật lại khi update_refine
charged = [1] * n_refine  # this array: to know which node in refined charging sequence requires charging (computed in the beginning of each round)

# print("Ước lượng đầu mỗi Cycle:")
# print("\tSigma_k: số round max ước lượng: {}".format(sigma_k))
# print("------------")
print("\tRefined charging sequence:", end=" ")
for index in refined_charging_sequence:
  print(index, end=" ")
# print("\n\tLength of RCS: {}".format(n_refine))
# print("-----------")

E_after_charging = E
E_l = np.array([E] * M)
index_last_node_charged_so_far = -1
total_time_before = 0
position = defaultdict(int)
round_min_num_of_rounds = [608] * M

# for round
start_cycle = time.time()
current_round = 0
finist_point = 10000
while(index_last_node_charged_so_far < finist_point):
  start_round = time.time()
  list_avai_MCs = find_index_of_avai_MCs(E_l)
  h_k_l = len(list_avai_MCs)
  p_k_l = min(h_k_l, n_refine - index_last_node_charged_so_far - 1)
  # print("------------")
  print("Thông số round {}:".format(current_round))
  # print("\tNhững MCs avai trong round {}, cùng năng lượng của chúng: ".format(current_round), end="")
  # for index in list_avai_MCs:
  #   print("{} - {}".format(index, E_l[index]),end=", ")
  # print("\n\th_k_l = {}, p_k_l = {}".format(h_k_l, p_k_l))
  # print("\n\tThông tin về dynamic_e_k khi bắt đầu round {} (chứa tất cả các Node):".format(current_round))
  # for i in range(len(dynamic_e_k)) :
  #   print("\t{}".format(dynamic_e_k[i]), end=' ')
  # plot_e_k.append(d)
  # print("\n\tNăng lượng của các MC khi bắt đầu round {} (đang xét E_l, có thể đã phải về BS rồi)): ".format(current_round), end=' ')
  # for i in E_l:
  #   print("{}".format(i), end=' ')
  # print("\n------------")

  _,_,_ = update_refine(E_l, e_k, total_time_before, index_last_node_charged_so_far, E_after_charging)

  def create_new_index_node():
    new_index_node = []   # độ dài p_k_l; được tính lại sau mỗi round; ánh xạ sang node ở trong L_K
    cnt = 0
    i = index_last_node_charged_so_far + 1
    while(i < n_refine):    # nhặt ra đủ p_k_l nodes trong RCS mà cần THỰC SỰ sạc
      if charged[i] == 1:
        new_index_node.append(refined_charging_sequence[i])
        cnt += 1
      else:
        print('Node {} không cần sạc'.format(refined_charging_sequence[i]),end=', ')
      i += 1
      if cnt == p_k_l: break 
    return new_index_node
  new_index_node = create_new_index_node()
  p_k_l = min(p_k_l, len(new_index_node))
  print("\tnew_index_node: ", new_index_node)
  # update p_k_l
  p_k_l = min(p_k_l, len(new_index_node))

  def create_new_index_MC():
    new_index_MC = []     # độ dài h_k_l; được tính lại sau mỗi round
    for j in range(h_k_l):
      new_index_MC.append(list_avai_MCs[j])
    return new_index_MC
  new_index_MC = create_new_index_MC()
  print("\tnew_index_MC: ", new_index_MC)
  # print("------------")

  def create_prev_pos_MC():
    prev_pos_MC = []    # gồm h_k_l phần từ
    for j in list_avai_MCs:
      if E_l[j] == E_after_charging:
        prev_pos_MC.append(N)
      else: 
        prev_pos_MC.append(position[j])
    return prev_pos_MC
  prev_pos_MC = create_prev_pos_MC()

  def create_t_l_max():
    actual_e_min = min(dynamic_e_k)
    # actual_e_min = min(e_k)
    actual_max_time = (E_MAX - actual_e_min) / p_r
    # actual_e_min = min(e_k)
    actual_max_time = (E_MAX - actual_e_min) / p_r
    return [actual_max_time] * M
  t_l_max = create_t_l_max()

  # print("\tINDEX_LAST_NODE_CHARGED_SO_FAR: {}".format(index_last_node_charged_so_far))
  index_last_node_charged_so_far, n_charged_before, n_uncharged_before = update_refine(E_l, e_k, total_time_before, index_last_node_charged_so_far, E_after_charging)
  # print("\tn_charged_before: {}, n_uncharged_before: {}".format(n_charged_before, n_uncharged_before))
  theta_k_l = calculate_time_threshold(E_l, total_time_before, current_round, n_charged_before, n_uncharged_before, E_after_charging, n_refine)
  print("\ttheta_k_l: {}".format(theta_k_l))
  # print("------------")

  def create_Elow_Ehigh():
    E_low = []
    E_high = []
    for i in new_index_node:    # i ở đây đang là index trong RCS
      e_low, e_high = calculate_E_i_low_and_E_i_high(E_l, i, sigma_k_next, e_k[refined_charging_sequence[i]], total_time_before, E)
      E_low.append(e_low)
      E_high.append(e_high)
    return E_low, E_high
  E_low, E_high = create_Elow_Ehigh()
  # print('\tE_low:', end=' ')
  # for el in E_low:
  #   print(el, end=', ')
  # print('\n\tE_high:', end=' ')
  # for eh in E_high:
  #   print(eh, end=', ')
  # print('\n')

  # giải
  
  omc = Omc(h_k_l, p_k_l, distance, e, p0, v, E_high, E_low, theta_k_l, p_r, prev_pos_MC, new_index_node, new_index_MC, dynamic_e_k)
  position, time_for_this_round, consumed, obj = omc.generic_algorithm(pop_size=100, max_generation=300)

  total_energy += obj
  round_energy.append(obj)

  # cập nhật sau khi giải
  E_l = np.array(update_after_MPSP(list_avai_MCs, E_l, consumed, E_after_charging))
  # update total time before
  total_time_before += time_for_this_round
  # print("Time for round {}: {}".format(current_round, time_for_this_round))
  # print("Total time so far: {}".format(total_time_before))
  print("TIME FOR ROUND {}: {}".format(current_round,  time.time()-start_round))
  current_round += 1
  
  # find finish point
  for i in range(n_refine):
    if charged[n_refine-i-1] == 1:
      finist_point = refined_charging_sequence[n_refine - i - 1]
      break
  print("\tINDEX_LAST_NODE_CHARGED_SO_FAR: {}".format(index_last_node_charged_so_far))
  print("Finish point: {}".format(finist_point))
  print("********GIẢI XONG ROUND*********")
  

print("Kết thúc một cycle")
# print("Thông tin về dynamic_e_k khi kết thúc round cuối {} (chứa tất cả các Node): ".format(current_round))
# for i in range(len(dynamic_e_k)):
#   print("\t{}".format(dynamic_e_k[i]), end=' ')
print("TIME: ", time.time()-start_cycle)

print("Total energy: ", total_energy)
print("Round energy:", round_energy)