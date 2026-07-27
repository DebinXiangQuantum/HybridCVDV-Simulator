OPENQASM 2.0;
include "qelib1.inc";
// canonical encoded-QASM surrogate for sc26_transfer_DVtoCV_nq2_c8
qreg q[5];
h q[0];
cx q[0],q[1];
h q[1];
cx q[1],q[2];
h q[2];
cx q[2],q[3];
h q[3];
cx q[3],q[4];
h q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
