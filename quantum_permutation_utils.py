###########################################################################################
                                        ## license ##
###########################################################################################
'''
MIT License

Copyright (c) 2022 pafloxy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

###########################################################################################
                                        ## imports ##
###########################################################################################



# from tkinter.tix import Tree
import numpy as np
from numpy import pi
import math
import seaborn as sns
from IPython.display import Image
import matplotlib.pyplot as plt
from typing import Mapping, Union, Iterable, Optional

from qiskit import *
from qiskit.circuit.library import *
from qiskit.algorithms import *
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit #, InstructionSet
from qiskit import quantum_info, IBMQ, Aer
from qiskit.quantum_info import partial_trace, Statevector, state_fidelity
from qiskit.utils import QuantumInstance
from qiskit.extensions import HamiltonianGate
from qiskit.circuit.quantumregister import Qubit
from qiskit.visualization import plot_histogram, plot_state_qsphere, plot_bloch_multivector, plot_bloch_vector

qsm = Aer.get_backend('qasm_simulator')
stv = Aer.get_backend('statevector_simulator')
aer = Aer.get_backend('aer_simulator')


##################################################################################################
                                    ## helper functions ##
##################################################################################################


def measure_and_plot(qc: QuantumCircuit, shots:int= 1024, show_counts:bool= False, return_counts:bool= False, return_plot:bool= True, measure_cntrls: bool = False, decimal_count_keys:bool = True , cntrl_specifier:Union[int, list, str] = 'all'):
    """ Measure and plot the state of the data registers, optionally measure the control ancillas, without modifying the original circuit.
        
        ARGS:
        ----
            qc : 'QuantumCircuit' 
                 the circuit to be measured

            shots: 'int' 
                    no. of shots for the measurement

            show_counts : 'bool' 
                           print the counts dictionary

            measure_cntrls : 'bool' 
                             indicates whether to measure the control ancilla registers.
            
            return_counts : 'bool'
                            returns the counts obtained ,  if True
            
            return_plot : 'bool'
                            returns the counts histogram plot obtained , if True

            measure_cntrls: 'bool'
                             indicates whether to measure the controll ancill qubits
                
            decimal_count_keys: 'bool'
                                if 'True' converts the binary state of the controll ancilllas to integer represntation

            cntrl_specifier : 'int' 
                                inidicates whihch of the control registers to meausure, 
                                for eg. cntrl_specifier= 2 refers to the first control ancilla cntrl_2
                                cntrl_specifier= 'all' refers to all the ancillas                                                           
        RETURNS:
        -------
            plots histogram over the computational basis states

     """
    qc_m = qc.copy()
    creg = ClassicalRegister( len(qc_m.qregs[0]) )
    qc_m.add_register(creg)
    qc_m.measure(qc_m.qregs[0], creg)

    if measure_cntrls== True:

        if isinstance(cntrl_specifier, int):
            print('int')##cflag
            if cntrl_specifier > len(qc_m.qregs) or cntrl_specifier < 1: raise ValueError(" 'cntrl_specifier' should be less than no. of control registers and greater than 0")

            creg_cntrl = ClassicalRegister(len(qc_m.qregs[cntrl_specifier-1]))
            qc_m.add_register(creg_cntrl)
            qc_m.measure(qc_m.qregs[cntrl_specifier-1], creg_cntrl )

        elif isinstance(cntrl_specifier, list ):
            print('list')##cflag
            for ancilla in cntrl_specifier:

                if ancilla > len(qc_m.qregs) or ancilla < 1: raise ValueError(" 'ancilla' should be less than no. of control registers and greater than 0")

                creg_cntrl = ClassicalRegister(len(qc_m.qregs[ancilla-1]))
                qc_m.add_register(creg_cntrl)
                qc_m.measure(qc_m.qregs[ancilla-1], creg_cntrl )

        elif isinstance(cntrl_specifier, str) and cntrl_specifier== "all":
            print('str')##cflag
            for reg in qc_m.qregs[1:] :
                creg = ClassicalRegister(len(reg))
                qc_m.add_register(creg)
                qc_m.measure(reg, creg)
                
    # plt.figure()
    counts = execute(qc_m, qsm, shots= shots).result().get_counts()
   
    if decimal_count_keys:
        counts_m = {}
        for key in counts.keys():
            # split_key = key.split(' ')
            key_m = int(key[::-1], 2)
            counts_m[key_m] = counts[key]
        counts = counts_m

    if show_counts== True: print(counts)
    
    if return_counts: return counts
    elif return_plot: return plot_histogram(counts)
    

   

##################################################################################################
                                 ## grover search sub-routines ##
##################################################################################################

def grover_retrival_op( encryption_key:dict, controls:list)-> QuantumCircuit:

    good_key = ''
    for cntrl in controls:
        if cntrl in encryption_key.keys():
            power = encryption_key[cntrl]
            n = len(cntrl)
            bin_power = format(power, 'b').zfill(n)
            good_key+= bin_power
        
        else :
            n = len(cntrl)
            bin_power = '0'*n
            good_key+= bin_power
    
    grover_op = grover([good_key])
    return grover_op


## initial state preparation ~
def s_psi0(p):
    """ Prepare a QuantumCircuit that intiates a state required
        input:
            p= amplitude 
        output:
            s_psi0 gate    """
            
    qc = QuantumCircuit(1, name= " S_psi0 ")
    theta = 2*np.arcsin(np.sqrt(p))
    qc.ry(theta, 0)

    return qc.to_gate() 



## string to oracle ~
def str_to_oracle(pattern: str, name= 'oracle', return_type = "QuantumCircuit" ) -> Union[QuantumCircuit,  Statevector] :
    """ Convert a given string to an oracle circuit
        ARGS:
        ----
             pattern: a numpy vector with binarry entries 
             return_type: ['QuantumCircuit', 'Statevector']

        RETURNS:
        ------- 
               'QuantumCircuit' implementing the oracle
               'Statevector' evolved under the action of the oracle   """


    l = len(pattern)
    qr = QuantumRegister(l, name='reg')
    a = AncillaRegister(1, name='ancilla')
    oracle_circuit = QuantumCircuit(qr, a, name= name+'_'+ pattern )
    for q in range(l):
        if(pattern[q]=='0'): oracle_circuit.x(qr[q])
    oracle_circuit.x(a)
    oracle_circuit.h(a)
    oracle_circuit.mcx(qr, a)
    oracle_circuit.h(a)
    oracle_circuit.x(a)
    for q in range(l):
        if(pattern[q]=='0'): oracle_circuit.x(qr[q])
    
    #oracle_circuit.barrier()
    if return_type == "QuantumCircuit":
        return oracle_circuit
    elif return_type == "Statevector":
        return Statevector.from_instruction(oracle_circuit)



## oracle prep ~
def generate_oracles(good_states: list) -> QuantumCircuit :
    """ Return a QuantumCircuit that implements the oracles given the good_states
        ARGS:
        ----
            good_states: list of good staes, states must be binary strings, Eg. ['00', '11']

        RETURNS:
        -------
           ' QuantumCircuit' iplementing the oracle circuits """

    oracles = [ str_to_oracle(good_state) for good_state in good_states ]
    oracle_circuit = oracles[0]
    for oracle in oracles[1:] :
        oracle_circuit.compose(oracle,  inplace= True)
    

    return oracle_circuit

## diffuser prep ~
def diffuser(l:int)-> QuantumCircuit :
    """ Generate the Diffuser operator for the case where the initial state  is 
        the equal superposition state of all basis vectors 

        ARGS:
        ----
            l: no. of data qubits
        
        RETURNS:
        -------
                QuantumCircuit

    """

    qr = QuantumRegister(l, name='reg')
    a = AncillaRegister(1, name='ancilla')
    circuit = QuantumCircuit(qr, a, name= 'Diff.')
    
    circuit.h(qr)
    circuit.x(qr)
    
    circuit.x(a)
    circuit.h(a)
    circuit.mcx(qr ,a)
    circuit.h(a)
    circuit.x(a)

    circuit.x(qr)
    circuit.h(qr)
          
    return circuit

## grover prep ~
def grover(good_states: list, insert_barrier:bool= False)-> QuantumCircuit:
    
    oracle_circuit = generate_oracles(good_states)

    num_qubits= oracle_circuit.num_qubits - oracle_circuit.num_ancillas
    diffuser_circuit = diffuser(num_qubits)
    if insert_barrier== True: oracle_circuit.barrier()
    oracle_circuit.compose(diffuser_circuit, inplace= True)

    return oracle_circuit


#############################################################
                                ## game utility ##
#############################################################

