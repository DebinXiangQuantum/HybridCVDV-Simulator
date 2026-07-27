OPENQASM 2.0;
include "qelib1.inc";
// canonical encoded-QASM surrogate for sc26_gkp_c4
qreg q[3];
ry(-2.9542792515501701) q[0];
rz(-1.4771396257750851) q[0];
ry(0.50490503043094259) q[1];
rz(0.2524525152154713) q[1];
ry(0.47402859661297114) q[2];
rz(0.23701429830648557) q[2];
cx q[0],q[1];
cx q[1],q[2];
