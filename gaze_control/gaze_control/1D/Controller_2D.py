import nengo
import numpy
import matplotlib.pyplot as plt
from Controller_1D import Controller_1D

class SBG_2D(nengo.Network):

    def __init__(self, n_neurons=50, inh_weight=-0.5, OPN_weight=-0.5, exc=0.005,
                 inh=0.005, hgain=1, vgain=1):

        # generate SC encoders in the only in the right side of unit sphere
        angles = numpy.linspace(numpy.pi / 2, 3 * numpy.pi / 2, neurons)
        x, y = numpy.cos(angles), numpy.sin(angles)
        SC_encoders = numpy.array([x, y])

        self.SC_left = nengo.Ensemble(n_neurons, dimensions=2)
        self.SC_right = nengo.Ensemble(n_neurons, dimensions=2)
        
        self.hor = SBG_Pair(exc=exc, inh=inh, gain=hgain, 
                                 inh_weight=inh_weight, include_OPN=False)
        self.ver = SBG_Pair(exc=exc, inh=inh, gain=vgain, 
                                 inh_weight=inh_weight, include_OPN=False)

        #population thats always inhibiting, but slows down to allow saccade execution
        #tuning curves have intercepts from 0 to 1 and increase to the left
        self.OPN = nengo.Ensemble(n_neurons, dimensions=1, 
                                  encoders=[[-1]] * neurons, 
                                  intercepts=numpy.linspace(0, -0.99, neurons))
        
        OPN_weights = [[OPN_weight] * neurons] * neurons

        #connections from OPN to EBN of both sides, OPN are shared
        nengo.Connection(self.OPN.neurons, self.hor.left.EBN.neurons, 
                         transform=OPN_weights, filter=inh)
        nengo.Connection(self.OPN.neurons, self.hor.right.EBN.neurons, 
                         transform=OPN_weights, filter=inh)
        
        #connections from OPN to EBN of both sides, OPN are shared
        nengo.Connection(self.OPN.neurons, self.ver.left.EBN.neurons, 
                         transform=OPN_weights, filter=inh)
        nengo.Connection(self.OPN.neurons, self.ver.right.EBN.neurons, 
                         transform=OPN_weights, filter=inh)

        def Rsin(x):
            return x[0] * numpy.sin((x[1] * (numpy.pi / 2)))

        def Rcos(x):
            return x[0] * numpy.cos((x[1] * (numpy.pi / 2)))

        #connect SC to LLBN
        nengo.Connection(self.SC_left, self.hor.right.LLBN, filter=exc, 
                         function=Rcos) #transform=[[1,0]]) 
        nengo.Connection(self.SC_right, self.hor.left.LLBN, filter=exc, 
                         function=Rcos) #transform=[[1,0]]) 
        nengo.Connection(self.SC_left, self.ver.right.LLBN, filter=exc, 
                         function=Rsin) #transform=[[0,1]]) 
        nengo.Connection(self.SC_right, self.ver.left.LLBN, filter=exc, 
                         function=Rsin) #transform=[[0,1]])

####################################################################################
#Test Code
####################################################################################

if __name__ == '__main__':

    model = nengo.Model("Eye Control")

    def sacc(t):                 
        return [0.4 * (t > 0.1) * (t < 0.38), 0.8 * (t > 0.1) * (t < 0.38)] 

    exc = 0.007
    inh = 0.05
    filter = 0.07

    with model:
        
        control = Controller_2D(exc=exc, inh=inh, inh_weight=-0.5, OPN_weight=-0.001, hgain=3 , vgain=3)

        #input
        sac_input = nengo.Node(output=sacc, label="Saccade Vector")

        #connect to left SC
        nengo.Connection(sac_input, control.SC_left, filter=None)

        #probes
        SC_left_p = nengo.Probe(control.SC_left, filter=exc)
        SC_right_p = nengo.Probe(control.SC_right, filter=exc)

        LLBN_hr_p = nengo.Probe(control.hor.right.LLBN, filter=exc)
        LLBN_hl_p = nengo.Probe(control.hor.left.LLBN, filter=exc)
        LLBN_vr_p = nengo.Probe(control.ver.right.LLBN, filter=exc)
        LLBN_vl_p = nengo.Probe(control.ver.left.LLBN, filter=exc)

        EBN_hr_p = nengo.Probe(control.hor.right.EBN, filter=exc)
        EBN_hl_p = nengo.Probe(control.hor.left.EBN, filter=exc)
        EBN_vr_p = nengo.Probe(control.ver.right.EBN, filter=exc)
        EBN_vl_p = nengo.Probe(control.ver.left.EBN, filter=exc)

        MN_hr_p = nengo.Probe(control.hor.right.MN, filter=filter)
        MN_hl_p = nengo.Probe(control.hor.left.MN, filter=filter)
        MN_vr_p = nengo.Probe(control.ver.right.MN, filter=filter)
        MN_vl_p = nengo.Probe(control.ver.left.MN, filter=filter)

        TN_hr_p = nengo.Probe(control.hor.right.TN, filter=exc)
        TN_hl_p = nengo.Probe(control.hor.left.TN, filter=exc)
        TN_vr_p = nengo.Probe(control.ver.right.TN, filter=exc)
        TN_vl_p = nengo.Probe(control.ver.left.TN, filter=exc)
    
    sim = nengo.Simulator(model, dt = 0.001, builder=nengo.builder.Builder(copy=False))
    sim.run(0.5)
    
    t = sim.trange()
    plt.figure()
    
    plt.subplot(542)
    plt.plot(t, sim.data(SC_left_p)[:,0], label='SC_left_hor')
    plt.legend(loc='best')
    plt.subplot(541)
    plt.plot(t, sim.data(SC_right_p)[:,0], label='SC_right_hor')
    plt.legend(loc='best')
    plt.subplot(544)
    plt.plot(t, sim.data(SC_left_p)[:,1], label='SC_left_ver')
    plt.legend(loc='best')
    plt.subplot(543)
    plt.plot(t, sim.data(SC_right_p)[:,1], label='SC_right_ver')
    plt.legend(loc='best')

    plt.subplot(545)
    plt.plot(t, sim.data(LLBN_hl_p), label='LLBN_left_hor')
    plt.legend(loc='best')
    plt.subplot(546)
    plt.plot(t, sim.data(LLBN_hr_p), label='LLBN_right_hor')
    plt.legend(loc='best')
    plt.subplot(547)
    plt.plot(t, sim.data(LLBN_vl_p), label='LLBN_left_ver')
    plt.legend(loc='best')
    plt.subplot(548)
    plt.plot(t, sim.data(LLBN_vr_p), label='LLBN_right_ver')
    plt.legend(loc='best')

    plt.subplot(549)
    plt.plot(t, sim.data(EBN_hl_p), label='EBN_left_hor')
    plt.legend(loc='best')
    plt.subplot(5,4,10)
    plt.plot(t, sim.data(EBN_hr_p), label='EBN_right_hor')
    plt.legend(loc='best')
    plt.subplot(5,4,11)
    plt.plot(t, sim.data(EBN_vl_p), label='EBN_left_ver')
    plt.legend(loc='best')
    plt.subplot(5,4,12)
    plt.plot(t, sim.data(EBN_vr_p), label='EBN_right_ver')
    plt.legend(loc='best')
    
    plt.subplot(5,4,13)
    plt.plot(t, sim.data(TN_hl_p), label='TN_left_hor')
    plt.legend(loc='best')
    plt.subplot(5,4,14)
    plt.plot(t, sim.data(TN_hr_p), label='TN_right_hor')
    plt.legend(loc='best')
    plt.subplot(5,4,15)
    plt.plot(t, sim.data(TN_vl_p), label='TN_left_ver')
    plt.legend(loc='best')
    plt.subplot(5,4,16)
    plt.plot(t, sim.data(TN_vr_p), label='TN_right_ver')
    plt.legend(loc='best')
    
    plt.subplot(5,4,17)
    plt.plot(t, sim.data(MN_hl_p), label='MN_left_hor')
    plt.legend(loc='best')
    plt.subplot(5,4,18)
    plt.plot(t, sim.data(MN_hr_p), label='MN_right_hor')
    plt.legend(loc='best')
    plt.subplot(5,4,19)
    plt.plot(t, sim.data(MN_vl_p), label='MN_left_ver')
    plt.legend(loc='best')
    plt.subplot(5,4,20)
    plt.plot(t, sim.data(MN_vr_p), label='MN_right_ver')
    plt.legend(loc='best')

    plt.figure()
    plt.plot(sim.data(MN_hr_p)[:,0], sim.data(MN_vr_p)[:,0])
    plt.gca().set_xlim([-0.1,1])
    plt.gca().set_ylim([-0.1,1])

    plt.show()




    
