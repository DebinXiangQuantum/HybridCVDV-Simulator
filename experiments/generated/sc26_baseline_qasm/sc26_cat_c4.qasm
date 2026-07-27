OPENQASM 2.0;
include "qelib1.inc";
// canonical encoded-QASM surrogate for sc26_cat_c4
qreg q[3];
ry(-1.6234086127561931) q[0];
rz(-0.81170430637809654) q[0];
ry(-1.4919404573361708) q[1];
rz(-0.74597022866808538) q[1];
ry(-0.99961506153015334) q[2];
rz(-0.49980753076507667) q[2];
cx q[0],q[1];
cx q[1],q[2];
