#Readme:
#This code demonstrates the numerical simualtion of the core of legged consensus control, which is converted from the robot computer.
#You can directly run this code and refer to the code comments to gain a deeper understanding of the corresponding equations in the paper.

# Note that this code content is not comprehensive and only includes the core algorithm in LCC: multi-feet consensus combined with sliding mode control.
# It does not include variable topography, foot trajectory planning and linear mapping, which will be fully open source in the future.

# You can change the parameter in line 186 to "self.display = 'velocity'" to view the velocity state of foot-end


import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
plt.ion()


class Agent():
    state=[0,0,0]
    state_=[0,0,0]
    T=0.025
    v_state=[0,0,0]
    u=[0,0,0]
    d_state=[0,0,0]
    d_v_state=[0,0,0]
    d_vv_state=[0,0,0]
    period=0
    state_list=[[] for i in range(3)]
    d_state_list=[[] for i in range(3)]
    period_list=[]
    Error=[0,0,0]
    r = [2, 2, 2]
    def __init__(self):
        self.state = [0, 0, 0]
        self.state_ = [0, 0, 0]
        self.v_state = [0, 0, 0]
        self.d_v_state=[1,1,1]
        self.d_vv_state=[1,1,1]
        self.u=[0,0,0]
        self.d_state = [0, 0, 0]
        self.period = 0
        self.state_list = [[] for i in range(3)]
        self.v_state_list = [[] for i in range(3)]
        self.d_state_list = [[] for i in range(3)]
        self.d_v_state_list = [[] for i in range(3)]
        self.period_list = []
        self.Error = [0, 0, 0]
        self.T = 0.025
        self.r = [2,2,2]
        self.v_r = [2,2,2]

    def r_update(self,print_flag=None):
        for i in range(3):
            if print_flag==True:
                print(2222,math.log(math.e,1+math.e**(-abs(self.state[i]-self.d_state[i]))),self.state[i],self.d_state[i])
            if self.state[i]-self.d_state[i]!=0 and self.v_state[i]-self.d_v_state[i]!=0:

                # Paper---Eq.(12)
                self.r[i]=4+4/(1+math.e**(-abs(self.state[i]-self.d_state[i])))
                self.v_r[i]=4+4*math.log(math.e,1+math.e**(-abs(self.state[i]-self.d_state[i]))) \
                    *(1+math.e**(-abs(self.state[i]-self.d_state[i])))*(-abs(self.v_state[i]-self.d_v_state[i]))
            else:
                self.r[i]=4
                self.v_r[i]=4

    def state_update(self,print_flag=None,type=True):
        if type==True:
            for i in range(3):
                self.d_state[i]=math.sin(self.period/2)
                self.d_v_state[i]=math.cos(self.period/2)/2
                self.d_vv_state[i]=-math.sin(self.period/2)/4

        for i in range(3):
            if print_flag==1 and i==0:
                print(1111,self.v_state[0],self.u[0])

            # Paper---Eq.(14)
            self.v_state[i]+=self.u[i]*self.T+self.Error[i]*self.T*0.02
            if print_flag==1 and i==0:
                print(1112,self.v_state[0])
            self.state[i]+=self.v_state[i]*self.T

            self.v_state[i]=(self.state[i]-self.state_[i])/self.T
            if print_flag==1 and i==0:
                print(1113,self.v_state[0])

            # Paper---Eq.(13)
            self.Error[i]+=self.d_state[i]-self.state[i]
            self.state_[i]=self.state[i]
            self.state_list[i].append(self.state[i])
            self.d_state_list[i].append(self.d_state[i])

            self.v_state_list[i].append(self.v_state[i])
            self.d_v_state_list[i].append(self.d_v_state[i])

        self.period += self.T
        self.period_list.append(self.period)
    def run(self):
        for i in range(3):
            self.d_state[i]=self.d_v_state[i]*self.period
            self.u[i]=(self.d_state[i]-self.state[i])*5+self.Error[i]*0.1+(self.d_v_state[i]-self.v_state[i])*5#PID
        for i in range(3):
            self.v_state[i]+=self.u[i]*self.T
            self.state[i]+=self.v_state[i]*self.T

            self.v_state[i]=(self.state[i]-self.state_[i])/self.T
            self.Error[i]+=self.d_state[i]-self.state[i]
            self.state_[i]=self.state[i]
            self.state_list[i].append(self.state[i])
            self.d_state_list[i].append(self.d_state[i])

            self.v_state_list[i].append(self.v_state[i])
            self.d_v_state_list[i].append(self.d_v_state[i])

        self.period += self.T
        self.period_list.append(self.period)
        plt.plot(self.period_list, self.state_list[0], color="red")
        plt.plot(self.period_list, self.d_state_list[0], color="blue")

        plt.grid()
        plt.show()
        plt.pause(0.001)

class Multi_Agent():
    agent0=Agent()
    agent1=Agent()
    agent2=Agent()
    agent3=Agent()
    agent4=Agent()
    agent5=Agent()
    agent=[agent0,agent1,agent2,agent3,agent4,agent5]

    stance_list=[0,1,2]
    swing_list=[3,4,5]
    N_stance=3
    N_swing=3
    N=[]

    State_Matrix=[[[],[]] for i in range(3)]
    Target_Matrix=[[[],[]] for i in range(3)]
    V_State_Matrix=[[[],[]] for i in range(3)]
    V_Target_Matrix=[[[],[]] for i in range(3)]
    VV_Target_Matrix=[[[],[]] for i in range(3)]

    L=[[[],[]] for i in range(3)]
    C=[[[],[]] for i in range(3)]
    R=[[[],[]] for i in range(3)]
    V_R=[[[],[]] for i in range(3)]
    U=[[[],[]] for i in range(3)]

    E_d_p = [[] for i in range(3)]
    E_d_v = [[] for i in range(3)]
    E_c_p = [[] for i in range(3)]
    E_c_v = [[] for i in range(3)]

    derta=[[[], []] for i in range(3)]

    Jeta_1 = [[[], []] for i in range(3)]
    Jeta_2 = [[[], []] for i in range(3)]
    Jeta_3 = [[[], []] for i in range(3)]
    Jeta_4 = [[[], []] for i in range(3)]
    Jeta_4_diag = [[[], []] for i in range(3)]

    q_1=1
    q_2=1
    q_3=1

    r=2
    seta=0.5
    a=1
    gamma=2
    test_flag=0
    rou=[[[],[]] for i in range(3)]
    plt.ion()
    fig = plt.figure()

    def __init__(self):
        self.stance_list = [0, 1, 2]
        self.swing_list = [3, 4, 5]
        self.N_stance=len(self.stance_list)
        self.N_swing=len(self.swing_list)
        self.N=[self.N_stance,self.N_swing]
        self.display = 'position'
        # self.display = 'velocity'
        for j in range(3):
            self.State_Matrix[j][0]=np.array([[self.agent[i].state[j] for o in range(self.N_stance)] for i in self.stance_list])
            self.State_Matrix[j][1]=np.array([[self.agent[i].state[j] for o in range(self.N_swing)] for i in self.swing_list])

            self.Target_Matrix[j][0] = np.array(
                [[self.agent[i].d_state[j] for o in range(self.N_stance)] for i in self.stance_list])
            self.Target_Matrix[j][1] = np.array(
                [[self.agent[i].d_state[j] for o in range(self.N_swing)] for i in self.swing_list])

            self.V_State_Matrix[j][0] = np.array(
                [[self.agent[i].v_state[j] for o in range(self.N_stance)] for i in self.stance_list])
            self.V_State_Matrix[j][1] = np.array(
                [[self.agent[i].v_state[j] for o in range(self.N_swing)] for i in self.swing_list])

            self.V_Target_Matrix[j][0] = np.array(
                [[self.agent[i].d_v_state[j] for o in range(self.N_stance)] for i in self.stance_list])
            self.V_Target_Matrix[j][1] = np.array(
                [[self.agent[i].d_v_state[j] for o in range(self.N_swing)] for i in self.swing_list])

            self.VV_Target_Matrix[j][0] = np.array(
                [[self.agent[i].d_vv_state[j] for o in range(self.N_stance)] for i in self.stance_list])
            self.VV_Target_Matrix[j][1] = np.array(
                [[self.agent[i].d_vv_state[j] for o in range(self.N_swing)] for i in self.swing_list])

            self.L[j][0]=np.ones([self.N_stance,self.N_stance])
            self.L[j][1] = np.ones([self.N_swing,self.N_swing])

            arr_1=np.array([-self.N_stance-1 for i in range(self.N_stance)])
            arr_2=np.array([-self.N_swing-1 for i in range(self.N_swing)])

            # Paper---Eq.(9,10)
            self.L[j][0]=self.L[j][0]+np.diag(arr_1)
            self.L[j][1]=self.L[j][1]+np.diag(arr_2)

            self.C[j][0]=np.ones([self.N_stance,self.N_stance])
            self.C[j][1]=np.ones([self.N_swing,self.N_swing])

            self.R[j][0]=np.diag([self.agent[i].r[j] for i in self.stance_list])
            self.R[j][1]=np.diag([self.agent[i].r[j] for i in self.swing_list])

            self.V_R[j][0] = np.diag([self.agent[i].v_r[j] for i in self.stance_list])
            self.V_R[j][1] = np.diag([self.agent[i].v_r[j] for i in self.swing_list])

            self.U[j][0]=np.diag([0 for i in self.stance_list])
            self.U[j][1]=np.diag([0 for i in self.swing_list])


        self.E_d_p = [[] for i in range(3)]
        self.E_d_v = [[] for i in range(3)]
        self.E_c_p = [[] for i in range(3)]
        self.E_c_v = [[] for i in range(3)]

        self.derta = [[[], []] for i in range(3)]

        self.Jeta_1 = [[[], []] for i in range(3)]
        self.Jeta_2 = [[[], []] for i in range(3)]
        self.Jeta_3 = [[[], []] for i in range(3)]
        self.Jeta_4 = [[[], []] for i in range(3)]
        self.Jeta_4_diag = [[[], []] for i in range(3)]

        self.q_0=100
        self.q_1 = 8
        self.q_2 = 1
        self.q_3 = 1

        self.r = 2
        self.seta = 0.5
        self.a = 1
        self.gamma = 2
        self.rou=[[[],[]] for i in range(3)]
        self.test_flag = 0
        self.swing_d_state=0
        self.stance_d_state=0
        self.swing_d_v_state=0
        self.stance_d_v_state=0
        self.swing_d_vv_state=0
        self.stance_d_vv_state=0


    def run(self):
        self.E_d_p = [[0 for j in range(6)] for i in range(3)]
        self.E_d_v = [[0 for j in range(6)] for i in range(3)]
        self.E_c_p = [[0 for j in range(6)] for i in range(3)]
        self.E_c_v = [[0 for j in range(6)] for i in range(3)]
        for i in range(6):
            self.agent[i].r_update()
        for j in range(3):
            for i in self.stance_list:
                self.E_d_p[j][i]=(self.agent[i].d_state[j]-self.agent[i].state[j])*self.agent[i].r[j]
                self.E_d_v[j][i] = (self.agent[i].d_v_state[j] - self.agent[i].v_state[j]) * self.agent[i].r[j] *self.seta
                for o in self.stance_list:
                    if o!=i:
                        self.E_c_p[j][i]+=(self.agent[o].state[j]-self.agent[i].state[j])*self.a
                        self.E_c_v[j][i]+=(self.agent[o].v_state[j]-self.agent[i].v_state[j])*self.a*self.gamma
                self.agent[i].u[j]=self.E_d_p[j][i]+self.E_d_v[j][i]+self.E_c_p[j][i]+self.E_c_v[j][i]
        for i in self.stance_list:
            self.agent[i].state_update()

        if abs(self.agent[0].period-5)<0.1 and self.test_flag==0:

            self.agent[1].state[0]+=0.2
            self.test_flag=1
        if self.display=='position':
            plt.plot(self.agent[0].period_list, self.agent[0].state_list[0], color="red")
            plt.plot(self.agent[0].period_list, self.agent[0].d_state_list[0], color="blue")

            plt.plot(self.agent[1].period_list, self.agent[1].state_list[0], color="purple")
            plt.plot(self.agent[1].period_list, self.agent[1].d_state_list[0], color="green")
        else:
            plt.plot(self.agent[0].period_list, self.agent[0].v_state_list[0], color="red")
            plt.plot(self.agent[0].period_list, self.agent[0].d_v_state_list[0], color="blue")

            plt.plot(self.agent[1].period_list, self.agent[1].v_state_list[0], color="purple")
            plt.plot(self.agent[1].period_list, self.agent[1].d_v_state_list[0], color="green")
        plt.grid()
        plt.show()
        plt.pause(0.01)


    def run_sliding_mode(self):
        self.E_d_p = [[[],[]] for i in range(3)]
        self.E_d_v = [[[],[]] for i in range(3)]
        self.E_c_p = [[[],[]] for i in range(3)]
        self.E_c_v = [[[],[]] for i in range(3)]
        self.derta=[[[], []] for i in range(3)]

        self.update()
        for i in range(6):
            if i==3:
                self.agent[i].r_update(print_flag=None)
            else:
                self.agent[i].r_update()

        for k in range(2):
            for j in range(3):

                #Paper---Eq.(20)
                self.E_c_p[j][k]=np.diag(np.diagonal(self.L[j][k]@(self.State_Matrix[j][k]-self.State_Matrix[j][k].T)))
                self.E_d_p[j][k]=self.R[j][k]@np.diag(np.diagonal(self.C[j][k]@(self.Target_Matrix[j][k].T-self.State_Matrix[j][k].T)))
                self.E_c_v[j][k]=np.diag(np.diagonal(self.gamma*self.L[j][k]@(self.V_State_Matrix[j][k]-self.V_State_Matrix[j][k].T)))
                self.E_d_v[j][k]=self.R[j][k]@np.diag(np.diagonal(self.seta*self.C[j][k]@(self.V_Target_Matrix[j][k].T-self.V_State_Matrix[j][k].T)))

                #Paper---Eq.(21)
                self.derta[j][k]=self.q_0*self.E_c_p[j][k]+self.q_1*self.E_d_p[j][k]+self.q_2*self.E_c_v[j][k]+self.q_3*self.E_d_v[j][k]

                #Paper---Eq.(30)
                self.Jeta_1[j][k]=self.q_2*self.gamma*self.L[j][k]
                self.Jeta_2[j][k]=-self.q_2*self.gamma*self.L[j][k]-self.q_3*self.seta*self.R[j][k]@self.C[j][k]
                self.Jeta_3[j][k]=self.gamma**(-1)*self.q_0*self.E_c_v[j][k]+self.seta**(-1)*self.q_1*self.E_d_v[j][k]+self.q_1*self.V_R[j][k]@np.linalg.inv(self.R[j][k])@self.E_d_p[j][k]\
                    +self.q_3*self.seta*self.R[j][k]@self.C[j][k]@(self.VV_Target_Matrix[j][k].T)+self.q_3*self.V_R[j][k]@np.linalg.inv(self.R[j][k])@self.E_d_v[j][k]

                # Paper---Eq.(32)
                self.Jeta_4[j][k]=-np.array(np.diag(np.diagonal(self.Jeta_3[j][k]))+self.rou[j][k]@sign(self.derta[j][k]))

                # Paper---Eq.(39)
                self.Jeta_4_diag[j][k]=np.transpose(np.array([np.diagonal(self.Jeta_4[j][k])]))

                # Paper---Eq.(40)
                G=np.array([self.Jeta_1[j][k][q]+np.ones([self.N[k]])@(np.array([self.Jeta_2[j][k][q]]).T)*np.diag([1 for i in range(self.N[k])])[q] for q in range(self.N[k])])

                # Paper---Eq.(37)
                self.U[j][k]=np.linalg.inv(G)@(self.Jeta_4_diag[j][k])

        for j in range(3):
            for i in range(6):
                if i in self.stance_list:
                    self.agent[i].u[j]=self.U[j][0][self.stance_list.index(i)][0]
                else:
                    self.agent[i].u[j]=self.U[j][1][self.swing_list.index(i)][0]

        for i in range(6):
            if i in self.stance_list:
                for j in range(3):
                    self.agent[i].d_state[j] = math.sin(self.agent[i].period / 2)
                    self.agent[i].d_v_state[j] = math.cos(self.agent[i].period / 2) / 2
                    self.agent[i].d_vv_state[j] = -math.sin(self.agent[i].period / 2) / 4
                self.stance_d_state = math.sin(self.agent[i].period / 2)
                self.stance_d_v_state = math.cos(self.agent[i].period / 2) / 2
                self.stance_d_vv_state = -math.sin(self.agent[i].period / 2) / 4
            else:
                for j in range(3):
                    self.agent[i].d_state[j] = math.sin(self.agent[i].period / 2+math.pi)
                    self.agent[i].d_v_state[j] = math.cos(self.agent[i].period / 2+math.pi) / 2
                    self.agent[i].d_vv_state[j] = -math.sin(self.agent[i].period / 2+math.pi) / 4
                self.swing_d_state = math.sin(self.agent[i].period / 2 + math.pi)
                self.swing_d_v_state = math.cos(self.agent[i].period / 2 + math.pi) / 2
                self.swing_d_vv_state = -math.sin(self.agent[i].period / 2 + math.pi) / 4
        for i in range(6):
            self.agent[i].state_update(type=None)

        plt.cla()
        if self.display == 'position':
            plt.plot(self.agent[0].period_list, self.agent[3].state_list[0], color="red")
            plt.plot(self.agent[0].period_list, self.agent[3].d_state_list[0], color="blue")

            plt.plot(self.agent[1].period_list, self.agent[4].state_list[0], color="purple")
            plt.plot(self.agent[1].period_list, self.agent[4].d_state_list[0], color="green")

            plt.xlabel(u'Time(s)', fontsize=14)
            plt.ylabel(u'X-axis position state(dm)', fontsize=14)
            plt.title(u"Multi-feet X-axis Position State", fontsize=14)
            plt.legend( ["No.3 foot actual state", "No.3 foot target state","No.4 foot actual state", "No.4 foot target state"], loc='upper right')
        if self.display=='velocity':
            plt.plot(self.agent[0].period_list, self.agent[3].v_state_list[0], color="red")
            plt.plot(self.agent[0].period_list, self.agent[3].d_v_state_list[0], color="blue")

            plt.plot(self.agent[1].period_list, self.agent[4].v_state_list[0], color="purple")
            plt.plot(self.agent[1].period_list, self.agent[4].d_v_state_list[0], color="green")

            plt.xlabel(u'Time(s)', fontsize=14)
            plt.ylabel(u'X-axis velocity state(mm/s)', fontsize=14)
            plt.title(u"Multi-feet X-axis Velocity State", fontsize=14)
            plt.legend(["No.3 foot actual state", "No.3 foot target state", "No.4 foot actual state",
                        "No.4 foot target state"], loc='upper right')

        plt.grid()

        plt.pause(0.01)



    def update(self):
        for j in range(3):
            self.State_Matrix[j][0]=np.array([[self.agent[i].state[j] for o in range(self.N_stance)] for i in self.stance_list])
            self.State_Matrix[j][1]=np.array([[self.agent[i].state[j] for o in range(self.N_swing)] for i in self.swing_list])

            self.Target_Matrix[j][0] = np.array(
                [[self.agent[i].d_state[j] for o in range(self.N_stance)] for i in self.stance_list])
            self.Target_Matrix[j][1] = np.array(
                [[self.agent[i].d_state[j] for o in range(self.N_swing)] for i in self.swing_list])

            self.V_State_Matrix[j][0] = np.array(
                [[self.agent[i].v_state[j] for o in range(self.N_stance)] for i in self.stance_list])
            self.V_State_Matrix[j][1] = np.array(
                [[self.agent[i].v_state[j] for o in range(self.N_swing)] for i in self.swing_list])

            self.V_Target_Matrix[j][0] = np.array(
                [[self.agent[i].d_v_state[j] for o in range(self.N_stance)] for i in self.stance_list])
            self.V_Target_Matrix[j][1] = np.array(
                [[self.agent[i].d_v_state[j] for o in range(self.N_swing)] for i in self.swing_list])

            self.VV_Target_Matrix[j][0] = np.array(
                [[self.agent[i].d_vv_state[j] for o in range(self.N_stance)] for i in self.stance_list])
            self.VV_Target_Matrix[j][1] = np.array(
                [[self.agent[i].d_vv_state[j] for o in range(self.N_swing)] for i in self.swing_list])

            self.R[j][0]=np.diag([self.agent[i].r[j] for i in self.stance_list])
            self.R[j][1]=np.diag([self.agent[i].r[j] for i in self.swing_list])

            # Paper---Eq.(25-3)
            self.V_R[j][0] = np.diag([self.agent[i].v_r[j] for i in self.stance_list])
            self.V_R[j][1] = np.diag([self.agent[i].v_r[j] for i in self.swing_list])

            self.L[j][0] = np.ones([self.N_stance, self.N_stance])
            self.L[j][1] = np.ones([self.N_swing, self.N_swing])

            arr_1 = np.array([-self.N_stance - 1 for i in range(self.N_stance)])
            arr_2 = np.array([-self.N_swing - 1 for i in range(self.N_swing)])

            # Paper---Eq.(9,10)
            self.L[j][0] = self.L[j][0] + np.diag(arr_1)
            self.L[j][1] = self.L[j][1] + np.diag(arr_2)

            self.C[j][0] = np.ones([self.N_stance, self.N_stance])
            self.C[j][1] = np.ones([self.N_swing, self.N_swing])

            self.U[j][0] = np.diag([0 for i in self.stance_list])
            self.U[j][1] = np.diag([0 for i in self.swing_list])


            parameter=[0 for i in range(6)]

            # Paper---Eq.(33)
            for i in range(6):
                if abs(self.agent[i].state[j] - self.agent[i].d_state[j]) >0.010:
                    parameter[i]=15
                else:
                    parameter[i]=0.1
            self.rou[j][0]=np.diag([parameter[i] for i in self.stance_list])
            self.rou[j][1]=np.diag([parameter[i] for i in self.swing_list])




def sign(x):
    y=np.diagonal(x)
    z=[]
    for i in range(len(y)):
        if y[i]>0:
            z.append(1)
        elif y[i]==0:
            z.append(0)
        else:
            z.append(-1)
    return np.diag(z)


multi_agent=Multi_Agent()


while(1):
    if abs(multi_agent.agent[0].period - 2) < 0.1 and multi_agent.test_flag == 0:
        multi_agent.agent[3].state[0] += 0.05 #Disturbance
        # multi_agent.agent[3].v_state[0] += 0.2 #Disturbance
        multi_agent.test_flag = 1
    multi_agent.run_sliding_mode()



