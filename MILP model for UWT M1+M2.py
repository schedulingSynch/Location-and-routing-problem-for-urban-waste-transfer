# -*- codeing: utf-8 -*-
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from dask.array import average
from docplex.mp.model import Model
import time
import copy


class LocationProblem:
    def __init__(self):
        self.mDir = ''
        self.mDir_input = 'input\\'
        self.mDir_solution = 'solution\\'
        self.mData = {}

    def read_data_M1(self, kx, nx, sx, sheetName1, sheetName2, fixcost):
        dataset1 = pd.read_excel(self.mDir + self.mDir_input + 'data_standard_ShanghaiChangning.xlsx', sheet_name=sheetName1)
        dataset2 = pd.read_excel(self.mDir + self.mDir_input + 'data_standard_ShanghaiChangning.xlsx', sheet_name=sheetName2)
        dataset3 = pd.read_excel(self.mDir + self.mDir_input + 'data_standard_ShanghaiChangning.xlsx', sheet_name="最终转运站")

        # dataset1 = pd.read_excel(self.mDir + self.mDir_input + 'data_standard.xlsx', sheet_name="小区")
        # dataset2 = pd.read_excel(self.mDir + self.mDir_input + 'data_standard.xlsx', sheet_name="小压站")
        # dataset3 = pd.read_excel(self.mDir + self.mDir_input + 'data_standard.xlsx', sheet_name="最终转运站")
        df1 = pd.DataFrame(dataset1)  ## 小区（垃圾收集点）
        df2 = pd.DataFrame(dataset2)  ## 小压站
        df3 = pd.DataFrame(dataset3)  ## 最终转运站

        K = list(df1['district'].values)[0: kx]
        K = [int(K[i]) for i in range(len(K))]
        KX = list(df1['KX'].values)[0: kx]
        KY = list(df1['KY'].values)[0: kx]
        D = list(df1['D'].values)[0: kx]  ### 小区k的垃圾总量
        D.insert(0, 0)
        KX.insert(0, 0)
        KY.insert(0, 0)

        N = list(df2['zipStation'].values)[0: nx]
        N = [int(N[i] + len(K)) for i in range(len(N))]         ### 小压站的序号, 在小区序号的基础上增序
        V = [0] * (len(K) + 1) + list(df2['V'].values)[0: nx]   ### 小压站的容量
        NX, NY = {}, {}
        N_x = list(df2['NX'].values)[0: nx]
        N_y = list(df2['NY'].values)[0: nx]
        for im in range(len(N)):
            NX[N[im]] = N_x[im]
            NY[N[im]] = N_y[im]

        B = [0] * (len(K) + 1) + list(df2[fixcost].values)

        XX = KX + N_x
        YY = KY + N_y

        S = list(df3['station'].values)[0: sx]
        S_x = list(df3['SX'].values)[0: sx]  ### 最终中转站的位置
        S_y = list(df3['SY'].values)[0: sx]  ### 最终中转站的位置
        p_x = list(df3['P'].values)[0: sx]  ### 最终中转站的容量
        SX, SY, P = {}, {}, {}
        s_ix = S[0]
        for ix in range(len(S_x)):
            SX[ix + s_ix] = S_x[ix]
            SY[ix + s_ix] = S_y[ix]

        C1 = np.zeros([len(XX) + 1, len(XX) + 1])
        for i in range(len(XX)):
            for j in range(len(XX)):
                if i == 0:
                    C1[i][j] = 0
                else:
                    C1[i][j] = 0.001 * np.sqrt((XX[i] - XX[j]) ** 2 + (YY[i] - YY[j]) ** 2)

        C2 = np.zeros([len(KX), len(NX)])
        J = set(K + N)  ## 小区 + 小压站
        Station = set(list(range(len(J) + 2, len(J) + len(S) + 2)))
        C3 = np.zeros([1 + list(NX.keys())[-1], len(J) + len(Station) + 3])
        for ix in range(len(S_x)):
            P[ix + len(J) + 2] = p_x[ix]
        for i_n in list(NX.keys()):  ## 小压站到最终转运站的量
            for js in Station:
                C3[i_n][js] = 0.001 * np.sqrt((XX[i_n] - SX[js-len(J) - 1]) ** 2 + (YY[i_n] - SY[js-len(J) - 1]) ** 2)

        return K, N, Station, D, V, P, C1, C2, C3, B

    def read_data_M2(self, kx, nx, sx, sheetName1, sheetName2, fixcost):
        dataset1 = pd.read_excel(self.mDir + self.mDir_input + 'data_standard_ShanghaiChangning.xlsx', sheet_name=sheetName1)
        dataset2 = pd.read_excel(self.mDir + self.mDir_input + 'data_standard_ShanghaiChangning.xlsx', sheet_name=sheetName2)
        dataset3 = pd.read_excel(self.mDir + self.mDir_input + 'data_standard_ShanghaiChangning.xlsx', sheet_name="最终转运站")
        #
        # dataset1 = pd.read_excel(self.mDir + self.mDir_input + 'data_standard.xlsx', sheet_name="小区")
        # dataset2 = pd.read_excel(self.mDir + self.mDir_input + 'data_standard.xlsx', sheet_name="小压站")
        # dataset3 = pd.read_excel(self.mDir + self.mDir_input + 'data_standard.xlsx', sheet_name="最终转运站")
        df1 = pd.DataFrame(dataset1)  ## 小区（垃圾收集点）
        df2 = pd.DataFrame(dataset2)  ## 小压站
        df3 = pd.DataFrame(dataset3)  ## 最终转运站

        K = list(df1['district'].values)[0: kx]
        K = [int(K[i]) for i in range(len(K))]
        KX = list(df1['KX'].values)[0: kx]
        KY = list(df1['KY'].values)[0: kx]
        D = list(df1['D'].values)[0: kx]  ### 小区k的垃圾总量
        D.insert(0, 0)
        KX.insert(0, 0)
        KY.insert(0, 0)

        N = list(df2['zipStation'].values)[0: nx]
        N = [int(N[i] + len(K)) for i in range(len(N))]  ### 小压站的序号, 在小区序号的基础上增序
        V = [0] * (len(K) + 1) + list(df2['V'].values)[0: nx]  ### 小压站的容量
        NX, NY = {}, {}
        N_x = list(df2['NX'].values)[0: nx]
        N_y = list(df2['NY'].values)[0: nx]
        for im in range(len(N)):
            NX[N[im]] = N_x[im]
            NY[N[im]] = N_y[im]

        B = [0] * (len(K) + 1) + list(df2[fixcost].values)

        S = list(df3['station'].values)[0: sx]
        S_x = list(df3['SX'].values)[0: sx]  ### 最终中转站的位置
        S_y = list(df3['SY'].values)[0: sx]  ### 最终中转站的位置
        p_x = list(df3['P'].values)[0: sx]  ### 最终中转站的容量
        SX, SY, P = {}, {}, {}
        s_ix = S[0]
        for ix in range(len(S_x)):
            SX[ix + s_ix] = S_x[ix]
            SY[ix + s_ix] = S_y[ix]


        """
        显示点的分布
        for i in range(6):
            plt.plot(NX[i], NY[i], '-or')
        for j in range(6,len(NX)):
            plt.plot(NX[j], NY[j], '-ok')
        for i in range(len(KX)):
            plt.plot(KX[i], KY[i], '-ob')
        plt.show()
        """

        XX = KX + N_x + S_x
        YY = KY + N_y + S_y

        C1 = np.zeros([len(XX) + 1, len(XX) + 1])
        for i in range(len(XX)):
            for j in range(len(XX)):
                if i == 0:
                    C1[i][j] = 0
                else:
                    C1[i][j] = 0.001 * np.sqrt((XX[i] - XX[j]) ** 2 + (YY[i] - YY[j]) ** 2)

        C2 = np.zeros([len(KX), len(NX)])

        J = set(K + N)  ## 小区 + 小压站
        Station = set(list(range(len(J) + 1, len(J) + len(S) + 1)))
        for ix in range(len(S_x)):
            P[ix + len(J) + 1] = p_x[ix]
        C3 = np.zeros([1 + list(NX.keys())[-1], len(J) + len(Station) + 1])
        for i in list(NX.keys()):
            for j in Station:
                C3[i][j] = 0.001 * np.sqrt((NX[i] - SX[j-len(J)]) ** 2 + (NY[i] - SY[j-len(J)]) ** 2)

        for ik in range(len(KX)):
            for js in Station:
                C3[ik][js] = 0.001 * np.sqrt((KX[ik] - SX[js-len(J)]) ** 2 + (KY[ik] - SY[js-len(J)]) ** 2)

        ## 构造需求(补充了压缩站和转运站的需求 0)
        nd = [0 for i in N]
        sd = [0 for i in Station]
        D = D + nd + sd + [0]

        return K, N, S, D, V, P, C1, C2, C3, B

    def solve_by_cplex_M1(self, K, N, Station, D, V, P, C1, C2, C3, B, num_vehicle, Q, beta, E1, E2, timeLimit):
        start = time.time()
        M1 = 1e+3  ### 一级转运的大M
        M2 = 2e+6  ### 二级转运的大M

        ### 小压站[按顺序]被启用
        ivx, V_need = 0, 0
        for vx in range(len(K) + 1, len(V)):
            ivx += 1
            V_need += V[vx]
            if V_need >= np.sum(D):
                break
            else:
                continue
        zipStationNum = len(K) + ivx  #### 需要小压站的数量
        print('启用小压站的数量：', range(len(K) + 1, zipStationNum+1))
        compactStations = list(range(len(K) + 1, zipStationNum+1))


        SP = {0}  ### 车辆的虚拟出发点
        J = set(K + N)  ## 小区 + 小压站
        SE = {len(J) + 1}  ### 车辆的虚拟终止点
        IS = SP | J
        IE = J | SE
        zipN = J - set(K)
        IK = {0} | set(K) ## 仅为小区
        ID = IS | IE

        m = Model("transferProblem")
        x = m.binary_var_matrix(K, zipN, name="x")   ### 小区k的垃圾被转运至小压站i
        z = m.binary_var_matrix(IS, IE, name="z")    ### 车辆依次访问小区 k + l
        h = m.integer_var_dict(ID, lb=0, name="h")   ### 车辆在小区k的载重量
        v = m.binary_var_dict(N, name='v')           ### 小压站被启用
        q = m.integer_var_matrix(K, zipN, name="q")  ### 小区k的垃圾被转运至小压站i的量
        y = m.binary_var_matrix(zipN, Station, name="y")   ### 小压站 i 压缩后的垃圾被二次转运至最终转运站 j
        u = m.integer_var_matrix(zipN, Station, name="u")  ### 小压站i的垃圾被二次转运至最终转运站j的量

        activateCost = m.sum(B[i] * v[i] for i in N)                                             ## 压缩站启用成本
        firstTransferCost = m.sum(C1[k1][k2] * E1 * z[k1, k2] for k1 in K for k2 in IE)  ## 第一级转运成本
        secondTransferCost = m.sum(C3[i, j] * E2 * u[i, j] for i in zipN for j in Station)             ## 第二级转运成本

        m.minimize(activateCost + firstTransferCost + secondTransferCost)   ### 总成本

        ### 分配约束
        m.add_constraints(m.sum(x[k, i] for i in zipN) == 1 for k in K)     ### 小区的垃圾被分配至小压站
        m.add_constraints(m.sum(q[k, i] for i in zipN) == D[k] for k in K)  ### 小区的垃圾全部被运往小压站
        m.add_constraints(m.sum(q[k, i] for k in K) <= V[i] for i in zipN)  ### 运往小压站i的垃圾总量不超过该站的容量V[i]
        m.add_constraints(q[k, i] <= M1 * x[k, i] for k in K for i in zipN)

        ### 第一级运输
        m.add_constraint(m.sum(z[0, k] for k in K) == num_vehicle)                  ### 车辆的起始点
        m.add_constraint(m.sum(z[k, e] for k in K for e in zipN) == num_vehicle)    ### 车辆的终止点
        m.add_constraints(m.sum(z[k1, k2] for k1 in IK) == 1 for k2 in K)  ### 垃圾收集点都要被访问 (入度)
        m.add_constraints(m.sum(z[k1, k2] for k2 in IE) == 1 for k1 in K)  ### 垃圾收集点都要被访问 (出度)
        m.add_constraints(z[k1, k1] == 0 for k1 in J)
        m.add_constraints(h[k2] >= h[k1] + D[k1] + (z[k1, k2] - 1) * Q for k1 in K for k2 in IE if k1 != k2)  ### 序约束
        m.add_constraints(h[k1] <= Q for k1 in K)

        m.add_constraints(m.sum(x[k, i] * D[k] for k in K) <= v[i] * V[i] for i in N)
        m.add_constraints(v[i] == 1 for i in range(len(K) + 1, zipStationNum + 1))      ## 若需要启用, 则令v_i=1.
        m.add_constraints(v[i] == 0 for i in range(zipStationNum + 1, len(J)))          ## 若不需要启用, 则令v_i=0
        m.add_constraints(v[i + 1] <= v[i] * M1 for i in range(len(K) + 1, len(J)))     ## 顺序启用约束
        m.add_constraints(m.sum(x[k, i] for k in K) >= v[i] for i in N)               ## (a) 启用小压站后, 才允许把小区的垃圾转运至此
        m.add_constraints(m.sum(x[k, i] for k in K) <= M1 * v[i] for i in N)          ## (b) 启用小压站后, 才允许把小区的垃圾转运至此
        m.add_constraints(z[k1, k2] <= v[k2] * M1 for k1 in K for k2 in zipN)         ## 只有压缩站被启用才允许分配小区

        ### 第二级转运
        m.add_constraints(m.sum(u[i, j] for j in Station) >= beta * m.sum(q[k, i] for k in K) for i in N)
        m.add_constraints(y[i, j] <= M2 * u[i, j] for i in N for j in Station)
        m.add_constraints(u[i, j] <= M2 * y[i, j] for i in N for j in Station)
        m.add_constraints(m.sum(u[i, j] for i in N) <= P[j] for j in Station)
        m.add_constraints(m.sum(y[i, j] for j in Station) >= v[i] for i in N)
        m.add_constraints(m.sum(y[i, j] for j in Station) <= M2 * v[i] for i in N)

        m.parameters.timelimit = timeLimit
        solution = m.solve(log_output=False)
        mip_gap = m.solve_details.mip_relative_gap

        R_ij, U = {}, {}
        if solution:
            # print("\nSolution:\n")
            end = time.perf_counter()
            duration = end - start
            # m.print_solution()  ## 输出决策的结果

            filename = self.mDir + self.mDir_solution + 'solution-M1.txt'
            file_handle = open(filename, 'w')
            file_handle.write("***********------***********\n")
            file_handle.write('objective and time_used is: \n[objective, time_used]\n')
            file_handle.write("[%s,%s]\n" % (solution.get_objective_value(), duration))

            file_handle.write("***********------***********\n")
            file_handle.write('The assignment relation of community, compacting station is as '
                              'follows: \n[community, compacting station, quantity]\n')
            for k in K:
                for i in N:
                    if x[k, i].solution_value > 0.9:
                        file_handle.write("[%s,%s,%s]\n" % (k, i, V[k]))

            file_handle.write("***********------***********\n")
            file_handle.write('The assignment relation of shelter and site '
                              'is as follows: \n[compacting station, terminal station, quantity]\n')
            for i in zipN:
                for j in Station:
                    if y[i, j].solution_value > 0.9:
                        file_handle.write("[%s,%s,%s]\n" % (i, j, u[i, j].solution_value))

            file_handle.write("***********------***********\n")
            file_handle.write('The total assigned quantity of shelter is as '
                              'follows: \n[shelter, total assigned quantity]\n')
            for k in K:
                for i in zipN:
                    if q[k, i].solution_value > 0.9:
                        file_handle.write("[%s,%s,%s]\n" % (k, i, q[k, i].solution_value))

            # print('route ****')
            for i1 in IK:
                for i2 in J:
                    if z[i1, i2].solution_value > 0.9:
                        R_ij[(i1, i2)] = 1

            for i in zipN:  ### 第二级转运过程转运的量
                for j in Station:
                    if y[i, j].solution_value > 0.9:
                        file_handle.write("[%s,%s,%s]\n" % (i, j, u[i, j].solution_value))
                        U[(i, j)] = u[i, j].solution_value


            file_handle.close()

        return compactStations, R_ij, U, np.ceil(activateCost.solution_value), mip_gap


    def reshapeSolution(self, fixedCost, R_ij, U, compactStations, Station, D, C1, C3, E1, E2, Vehicle):
        start = 0
        succ = {}
        iCost1, iCost2 = 0, 0
        for (i, j), val in R_ij.items():
            if val == 1:
                succ.setdefault(i, []).append(j)
        route = {}
        rid = 1

        for nxt in succ.get(start, []):# 2) 从 start 的每个后继分别出发生成一条路径
            path = [start, nxt]
            cur = nxt
            visited = set([start])  # 防止环
            while cur not in compactStations:
                if cur in visited:
                    break
                visited.add(cur)
                if cur not in succ or len(succ[cur]) == 0: # 若没有后继则断链结束
                    break
                cur = succ[cur][0] # 若有多个后继，这里默认取第一个（如需分支可拓展）
                path.append(cur)
            route[rid] = path
            rid += 1

        H = {}
        for ir in route.keys():
            iloaded = 0
            for ie in range(len(route[ir])-1):
                iloaded += D[route[ir][ie]]
                H[(ir, route[ir][ie])] = iloaded
        print(route)

        iCost1, iCost2 = 0, 0
        ### *** 第一级转运成本 (车辆启用+转运)
        vehicleCost1 = Vehicle['C1'] * len(route)  ## 启用成本
        for ir in route.keys():
            i_c_x = 0
            for ix in range(len(route[ir]) - 1):
                i_c_x += C1[route[ir][ix]][route[ir][ix + 1]] * E1 * H[(ir, route[ir][ix])]
            iCost1 += i_c_x
        # print('firstTransferCost:', np.ceil(iCost1))

        ## *** 第二级转运成本(车辆启用+配送)
        secondLoading = 0
        for u in U.keys():
            iCost2 += C3[u[0]][u[1]] * E2 * U[u]
            secondLoading += int(U[u])
        vehicleCost2 =  int(np.ceil(secondLoading / Vehicle['Q2'])) * Vehicle['C2']  ## 启用成本

        # print('secondTransferCost:', np.ceil(iCost2))
        # print('activateCost:', fixedCost)
        # print('totalCost: ', np.ceil(iCost2 + iCost1 + fixedCost))

        return iCost1, iCost2, fixedCost, vehicleCost1, vehicleCost2


    def solve_by_cplex_M2(self, K, N, S, D, V, P, C1, C2, C3, B, num_vehicle, Q, beta, E1, E2, E3, timeLimit):
        start = time.time()
        M1 = 1e+3  ### 一级转运的大M
        M2 = 2e+6  ### 二级转运的大M

        SP = {0}  ### 车辆的虚拟出发点
        J = set(K + N)  ## 小区 + 小压站
        SE = {len(J) + len(S) + 1}  ### 车辆的虚拟终止点
        Station = set(list(range(len(J) + 1, len(J) + len(S) + 1)))
        IS = SP | J
        IE = J | Station | SE
        zipN = J - set(K)
        IK = {0} | set(K) ## 仅为小区
        ID = IS | IE
        NS = zipN | Station ### 压缩站+最终转运站

        m = Model("transferProblem")
        x = m.binary_var_matrix(K, NS, name="x")        ### 小区k的垃圾被转运至小压站i
        z = m.binary_var_matrix(IS, IE, name="z")       ### 车辆依次访问小区 k + l
        h = m.integer_var_dict(ID, lb=0, name="h")      ### 车辆在小区k的载重量
        p = m.integer_var_dict(ID, lb=0, name="p")      ### 线性化辅助变量 h_a * z_ab
        v = m.binary_var_dict(N, name='v')              ### 小压站被启用
        q = m.integer_var_matrix(K, NS, name="q")       ### 小区k的垃圾被转运至小压站i的量
        y = m.binary_var_matrix(zipN, Station, name="y")    ### 小压站 i 压缩后的垃圾被二次转运至最终转运站 j
        u = m.integer_var_matrix(zipN, Station, name="u")   ### 小压站i的垃圾被二次转运至最终转运站j的量
        # TODO:
        w = m.binary_var_cube(range(num_vehicle), IS, IE, name="w")

        activateCost = m.sum(B[i] * v[i] for i in N)                                            ## 压缩站启用成本
        firstTransferCost = m.sum(C1[k1][k2] * E1 * D[k1] * z[k1, k2] for k1 in K for k2 in IE)  ## 第一级转运成本
        # firstTransferCost = m.sum(C1[k1][k2] * E1 * D[k1] * h[k1] for k1 in K for k2 in IE)  ## 第一级转运成本
        secondTransferCost = m.sum(C3[i, j] * E2 * u[i, j] for i in zipN for j in Station)             ## 第二级转运成本
        directTransferCost = m.sum(C1[k][j] * E3 * D[k] * x[k, j] for k in K for j in Station)   ## 直接转运成本

        m.minimize(activateCost + firstTransferCost + secondTransferCost + directTransferCost)   ### 总成本

        ### 线性化约束 h_a * z_ab
        # m.add_constraints(p[a] >= h[a] + Q * (z[a, b] - 1) for a in K for b in IE)  ### 线性化约束(1)
        # m.add_constraints(p[a] <= h[a] for a in K)                                  ### 线性化约束(2)

        # m.minimize(m.sum(B[i] * v[i] for i in N) +
        #            m.sum(C1[k1][k2] * E1 * D[k1] * z[k1, k2] for k1 in K for k2 in IE) +
        #            m.sum(C3[i, j] * E2 * u[i, j] for i in zipN for j in Station)+
        #            m.sum(C1[k][j] * E3 * D[k] * x[k, j] for k in K for j in Station))
        #
        ### 分配约束
        m.add_constraints(m.sum(x[k, i] for i in zipN) + m.sum(x[k, j] for j in Station) == 1 for k in K)  ### 小区的垃圾被分配至小压站 或 最终转运站
        m.add_constraints(m.sum(q[k, i] for i in NS) == D[k] for k in K)  ### 小区的垃圾全部被运往小压站 或 直接转运至 最终转运站
        m.add_constraints(m.sum(q[k, i] for k in K) <= V[i] for i in zipN)  ### 运往小压站i的垃圾总量不超过该站的容量V[i]
        m.add_constraints(q[k, i] <= M1 * x[k, i] for k in K for i in NS)

        ### 第一级运输
        #* m.add_constraint(m.sum(z[0, k] for k in K) == num_vehicle)  ### 车辆的起始点
        #* m.add_constraint(m.sum(z[k, e] for k in K for e in NS) == num_vehicle)  ### 车辆的终止点
        m.add_constraints(m.sum(z[k1, k2] for k1 in IK) == 1 for k2 in K)  ### 垃圾收集点都要被访问 (入度)
        m.add_constraints(m.sum(z[k1, k2] for k2 in IE) == 1 for k1 in K)  ### 垃圾收集点都要被访问 (出度)
        m.add_constraints(z[k1, k1] == 0 for k1 in J)
        m.add_constraints(h[k2] >= h[k1] + D[k1] + (z[k1, k2] - 1) * Q for k1 in K for k2 in IE if k1 != k2)  ### 序约束
        m.add_constraints(h[k1] <= Q for k1 in K)

        m.add_constraints(m.sum(w[r, k1, k2] for k1 in IS) == m.sum(w[r, k2, k1] for k1 in IE) for k2 in K for r in range(num_vehicle))
        m.add_constraints(m.sum(w[r, k1, k2] for k2 in IE for r in range(num_vehicle)) == 1 for k1 in K)
        m.add_constraints(m.sum(w[r, 0, k2] for k2 in IE) == 1 for r in range(num_vehicle))
        m.add_constraints(z[k1, k2] == m.sum(w[r, k1, k2] for r in range(num_vehicle)) for k1 in IS for k2 in IE)

        m.add_constraints(z[k, j] <= M2 * x[k, j] for k in K for j in Station)
        m.add_constraints(z[k, j] >= x[k, j] for k in K for j in Station)


        ### 小压站[按顺序]被启用
        ivx, V_need = 0, 0
        for vx in range(len(K) + 1, len(V)):
            ivx += 1
            V_need += V[vx]
            if V_need >= np.sum(D):
                break
            else:
                continue
        zipStationNum = len(K) + ivx  #### 需要小压站的数量
        print('启用小压站的数量：', range(len(K) + 1, zipStationNum))
        m.add_constraints(m.sum(x[k, i] * D[k] for k in K) <= v[i] * V[i] for i in N)
        m.add_constraints(v[i] == 1 for i in range(len(K) + 1, zipStationNum + 1))  ## 若需要启用, 则令v_i=1.
        m.add_constraints(v[i + 1] <= v[i] for i in range(len(K) + 1, len(J)))      ## 顺序启用约束
        m.add_constraints(m.sum(x[k, i] for k in K) >= v[i] for i in N)             ## (a) 启用小压站后, 才允许把小区的垃圾转运至此
        m.add_constraints(m.sum(x[k, i] for k in K) <= M1 * v[i] for i in N)        ## (b) 启用小压站后, 才允许把小区的垃圾转运至此
        m.add_constraints(z[k1, k2] <= v[k2] * M1 for k1 in K for k2 in zipN)       ## 只有压缩站被启用才允许分配小区

        ### 第二级转运
        m.add_constraints(m.sum(u[i, j] for j in Station) >= beta * m.sum(q[k, i] for k in K) for i in N)
        m.add_constraints(y[i, j] <= M2 * u[i, j] for i in N for j in Station)
        m.add_constraints(u[i, j] <= M2 * y[i, j] for i in N for j in Station)
        m.add_constraints(m.sum(u[i, j] for i in N) + m.sum(q[k, j] for k in K)<= P[j] for j in Station)
        m.add_constraints(m.sum(q[k, j] for k in K)<= P[j] for j in Station)
        m.add_constraints(m.sum(y[i, j] for j in Station) >= v[i] for i in N)
        m.add_constraints(m.sum(y[i, j] for j in Station) <= M2 * v[i] for i in N)

        m.parameters.timelimit = timeLimit
        solution = m.solve(log_output=True)
        mip_gap = m.solve_details.mip_relative_gap

        X, U = {}, {}
        route = {}
        if solution:
            # print("\nSolution:\n")
            # print('objective: ', solution.get_objective_value())
            end = time.perf_counter()
            duration = end - start
            # m.print_solution()  ## 输出决策的结果

            filename = self.mDir + self.mDir_solution + 'solution-M2.txt'
            file_handle = open(filename, 'w')
            file_handle.write("***********------***********\n")
            file_handle.write('objective and time_used is: \n[objective, time_used]\n')
            file_handle.write("[%s,%s]\n" % (solution.get_objective_value(), duration))

            for k in K:
                for i in N:
                    if x[k, i].solution_value > 0.9:
                        file_handle.write("[%s,%s,%s]\n" % (k, i, V[k]))
                        X[k] = i

            file_handle.write("***********------***********\n")
            file_handle.write('The assignment relation of shelter and site '
                              'is as follows: \n[compacting station, terminal station, quantity]\n')
            for i in zipN:      ### 第二级转运过程转运的量
                for j in Station:
                    if y[i, j].solution_value > 0.9:
                        file_handle.write("[%s,%s,%s]\n" % (i, j, u[i, j].solution_value))
                        U[(i, j)] = u[i, j].solution_value

            file_handle.write("***********------***********\n")
            file_handle.write('The total assigned quantity of shelter is as '
                              'follows: \n[shelter, total assigned quantity]\n')
            for k in K:
                for i in NS:
                    if q[k, i].solution_value > 0.9:
                        file_handle.write("[%s,%s,%s]\n" % (k, i, q[k, i].solution_value))

            # print('**** 车辆访问顺序 ***** w[r, a, b]')
            for r in range(num_vehicle):
                list_i = []
                list_j = []
                list_b = [0]
                for i in IS:
                    for j in IE:
                        if w[r, i, j].solution_value > 0.5:
                            list_i.append(i)
                            list_j.append(j)
                            # print(r, i, j)
                for l in range(len(list_i)):
                    ds = list_i.index(list_b[-1])
                    list_b.append(list_j[ds])
                route[r] = list_b

            file_handle.close()


        return X, U, J, N, Station, route, np.ceil(activateCost.solution_value), mip_gap

    def totalCostReShape(self, fixedCost, route, X, U, C1, C3, D, J, N, Station, E1, E2, E3, Vehicle):
        reshapeRoute = copy.deepcopy(route)
        ### 计算路径各节点的装载量
        H = {}
        for ir in route.keys():
            iloaded = 0
            if route[ir][-1] not in Station:
                reshapeRoute[ir].insert(-1, X[route[ir][-2]]) ## 构造新序列,把压缩站插入到序列中
            for ie in route[ir]:
                iloaded += D[ie]
                H[(ir, ie)] = iloaded
        print(route)

        ## *** 第一级 + 直接 转运成本 (车辆启用+转运)
        iCost1, iCost3 = 0, 0
        for ir in route.keys():
            i_c_x, i_c_d = 0, 0
            if route[ir][-1] not in Station: ## 前往compacting station的车辆
                for ix in range(len(reshapeRoute[ir]) - 1):
                    i_c_x += C1[reshapeRoute[ir][ix]][reshapeRoute[ir][ix + 1]] * E1 * H[(ir, route[ir][ix])]
            else:
                for ix in range(len(reshapeRoute[ir]) - 2):  #### 前往Terminal Station的车辆
                    i_c_d += C1[reshapeRoute[ir][ix]][reshapeRoute[ir][ix + 1]] * E1 * H[(ir, route[ir][ix])]
                i_c_d += C3[reshapeRoute[ir][-2]][reshapeRoute[ir][-1]] * E3 * H[(ir, route[ir][-2])]

            iCost1 += i_c_x
            iCost3 += i_c_d

        vehicleCost1 = Vehicle['C1'] * len(route)  ## 启用成本
        # print('firstTransferCost:', np.ceil(iCost1))


        ## *** 第二级转运成本
        iCost2, secondLoading = 0, 0
        if iCost1 <= 1e-3:
            # print('secondTransferCost:', np.ceil(iCost2))
            fixedCost = 0
            # print('activateCost:', 0)
        else:
            for u in U.keys():
                iCost2 += C3[u[0]][u[1]] * E2 * U[u]
                secondLoading += int(U[u])
        vehicleCost2 = int(np.ceil(secondLoading / Vehicle['Q2'])) * Vehicle['C2']  ## 启用成本
        #     print('secondTransferCost:', np.ceil(iCost2))
        #     print('activateCost:', fixedCost)
        #
        # print('directTransferCost:', np.ceil(iCost3))
        # print('totalCost: ', np.ceil(iCost3 + iCost2 + iCost1 + fixedCost))


        return iCost1, iCost2, iCost3, fixedCost, vehicleCost1, vehicleCost2

    def experiment_MILP_M1(self, kx, beta, E1, E2, sheetName1, fixcost):
        cpu_start = time.time()
        zipStationSet = {'N50': 2, 'N100': 4,'N150': 5, 'N200-EXP3': 9, 'N250': 5, 'N300': 7}
        Vehicle = {'C1': 300, 'C2': 1000, 'Q1': 20, 'Q2': 50}  ###***  车辆装载量及启用成本

        sheetName2 = 'N200-EXP3'             ###***  小压站集合的 sheet
        nx = zipStationSet[sheetName2]  ## 小型压缩站数量 （包括冗余的）  ### for CZQ: 这个小压站数量，要随算例改变而改变，否则可能会报错
        sx = 1  ### 最终转运站数量
        Q = 20  ### 第一次转运车辆的容量
        timeLimit = 1800  ###*** 代码的最大计算时间
        K, N, Station, D, V, P, C1, C2, C3, B = self.read_data_M1(kx, nx, sx, sheetName1, sheetName2, fixcost)
        num_vehicle = int(np.ceil(np.sum(D) / Q) + 0)  ### 车辆数量
        print("车辆数与实际车辆数：", np.ceil(np.sum(D) / Q), '\t', num_vehicle)
        compactStations, R_ij, U, fixedCost, mip_gap = self.solve_by_cplex_M1(K, N, Station, D, V, P, C1, C2, C3, B,
                                                                              num_vehicle, Q, beta, E1, E2, timeLimit)
        iCost1, iCost2, fCost, vehicleCost1, vehicleCost2 = self.reshapeSolution(fixedCost, R_ij, U, compactStations,
                                                                                 Station, D, C1, C3, E1, E2, Vehicle)
        tCost = np.ceil(iCost1 + iCost2 + fCost + vehicleCost1 + vehicleCost2)
        cpu = time.time() - cpu_start

        print("M1: K = {:2} | N = {:2} | beta = {:} | tC = {:7.1f} | iC1 = {:7.1f} | iC2 = {:7.1f} | iCf = {:7.1f} | "
              "openC1 = {:5.1f} | openC2 = {:5.1f} | CPU = {:5.3f} secs | gap {:5.4f}".format(len(K), len(N), beta,
                                                                                              tCost, iCost1, iCost2,
                                                                                              fCost, vehicleCost1,
                                                                                              vehicleCost2, cpu, mip_gap))

        return len(K), len(N), beta, tCost, iCost1, iCost2, fCost, vehicleCost1, vehicleCost2, cpu, mip_gap

    def experiment_MILP_M2(self, kx, beta, E1, E2, E3, sheetName1, fixcost):
        cpu_start = time.time()
        zipStationSet = {'N50': 2, 'N100': 4,'N150': 6, 'N200-EXP3': 9, 'N250': 5, 'N300': 7}
        Vehicle = {'C1': 300, 'C2': 1000, 'Q1': 20, 'Q2': 50}  ###***  车辆装载量及启用成本


        sheetName2 = 'N200-EXP3'               ###***  小压站集合的 sheet
        nx = zipStationSet[sheetName2]    ## 小型压缩站数量 （包括冗余的）
        sx = 1  ### 最终转运站数量
        Q = 20  ### 车辆的容量
        timeLimit = 1800     ###*** 代码的最大计算时间
        # E1, E2, E3 = 0.05, 0.18, 0.22    ###*** 第一级转运的单位距离单位重量成本, 第二级转运单位距离单位重量成本, 直接转运至转运中心的单位距离单位重量成本
        K, N, S, D, V, P, C1, C2, C3, B = self.read_data_M2(kx, nx, sx, sheetName1, sheetName2, fixcost)
        num_vehicle = int(np.ceil(np.sum(D) / Q) + 50)   ### 车辆数量
        print("车辆数与实际车辆数：", np.ceil(np.sum(D) / Q), '\t', num_vehicle)
        X, U, J, N, Station, route, fixedCost, mip_gap = self.solve_by_cplex_M2(K, N, S, D, V, P, C1, C2, C3, B,
                                                                                num_vehicle, Q, beta, E1, E2, E3, timeLimit)
        iCost1, iCost2, iCost3, fCost, vehicleCost1, vehicleCost2 = self.totalCostReShape(fixedCost, route, X, U, C1,
                                                                                          C3, D, J, N, Station, E1, E2, E3, Vehicle)
        tCost = np.ceil(iCost1 + iCost2 + iCost3 + fCost + vehicleCost1 + vehicleCost2)
        cpu = time.time() - cpu_start
        print("M2: K = {:2} | N = {:2} | beta = {:} | tC = {:7.1f} | iC1 = {:7.1f} | iC2 = {:7.1f} | iC3 = {:7.1f} | i"
              "Cf = {:7.1f} | openC1 = {:5.1f} | openC2 = {:5.1f} | CPU = {:5.3f} secs | gap {:5.4f}".format(len(K),
                                                                                                             len(N), beta, tCost,
                                                                                                             iCost1, iCost2, iCost3,
                                                                                                             fCost, vehicleCost1,
                                                                                                             vehicleCost2, cpu, mip_gap))

        return len(K), len(N), beta, tCost, iCost1, iCost2, iCost3, fCost, vehicleCost1, vehicleCost2, cpu, mip_gap

if __name__ == '__main__':
    a = LocationProblem()
    K_X = [10, 20, 30, 40, 50, 100, 150, 200]       ###**** 废弃物清运点数量
    E1, E2, E3 = 9, 18, 12          ###**** 每一个箱子（约180Kg）每公里的成本(可调整)
    beta = 0.8                      ###**** 小压站的压缩比
    rSolution1, rSolution2 = {}, {}

    ################ EXP2 在200个点中随机取相应数量的点  计算   ##########################
    ################ 模型 M1  ***********
    sheetName1 = 'EXP3'
    columns1 = ["Model", "K", "N", "beta", "tC", "iC1", "iC2", "iCf", "openC1", "openC2", "CPU", "gap"]
    # F_COST = ['cost50', 'cost75', 'cost0', 'cost125', 'cost150']
    F_COST = ['cost0']
    # MM1 = 'M2'
    for kx in K_X:
        for fixcost in F_COST:
            kmax1, Nmax1, beta1, tCost, iCost1, iCost2, fCost, vehicleCost1, vehicleCost2, cpu, mip_gap = a.experiment_MILP_M1(kx, beta, E1, E2, sheetName1, fixcost)      ### 模型一
            case_name = f"{sheetName1}_fixC{fixcost}_k{kx}"
            rSolution1[case_name] = ['M1', kmax1, Nmax1, beta1, tCost, iCost1, iCost2, fCost, vehicleCost1, vehicleCost2, cpu, mip_gap]
    result_df1 = pd.DataFrame.from_dict(rSolution1, orient="index", columns=columns1)
    result_df1 = result_df1.reset_index()
    result_df1 = result_df1.rename(columns={"index": "Case"})
    output_file1 =  "EXP3_M1_results.xlsx"
    result_df1.to_excel("D:\\yzwang\\HZH\\00-HZH_lab\\05-程源哲-废弃物管理所\\01-论文\\01-城市废弃物清运小压站配置与两级转运路径优化\\solution\\" + output_file1, index=False)
    # result_df1.to_excel("D:\\yzwang_Station\\Experiments\\6-4-Routing in urban logistices\\comps\\solution\\" + output_file1, index=False)


    ################### 模型M2  ************
    sheetName1 = 'EXP3'
    columns2 = ["Model", "K", "N", "beta", "tC", "iC1", "iC2", "dC", "iCf", "openC1", "openC2", "CPU", "gap"]
    # vehicleAssment = {'50': 300, '100': 1000, '150': 20, '200': 50}  ###***  车辆实际使用数量
    F_COST = ['cost50', 'cost75', 'cost0', 'cost125', 'cost150']
    for kx in K_X:
        for fixcost in F_COST:
            kmax2, Nmax2, beta2, tCost, iCost1, iCost2, dCost, fCost, vehicleCost1, vehicleCost2, cpu, mip_gap = a.experiment_MILP_M2(kx, beta, E1, E2, E3, sheetName1, fixcost)  ### 模型二
            case_name = f"{sheetName1}_fixC{fixcost}_k{kx}"
            rSolution2[case_name] = ['M2', kmax2, Nmax2, beta2, tCost, iCost1, iCost2, dCost, fCost, vehicleCost1, vehicleCost2, cpu, mip_gap]
    result_df2 = pd.DataFrame.from_dict(rSolution2, orient="index", columns=columns2)
    result_df2 = result_df2.reset_index()
    result_df2 = result_df2.rename(columns={"index": "Case"})
    output_file2 = "EXP3_M2_results.xlsx"
    result_df2.to_excel("D:\\yzwang\\HZH\\00-HZH_lab\\05-程源哲-废弃物管理所\\01-论文\\01-城市废弃物清运小压站配置与两级转运路径优化\\solution\\" + output_file2, index=False)
    # result_df2.to_excel("D:\\yzwang_Station\\Experiments\\6-4-Routing in urban logistices\\comps\\solution\\" + output_file2, index=False)

