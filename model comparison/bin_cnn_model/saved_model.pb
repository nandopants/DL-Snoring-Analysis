��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
�
SGD/dense_7/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameSGD/dense_7/bias/momentum
�
-SGD/dense_7/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_7/bias/momentum*
_output_shapes
:*
dtype0
�
SGD/dense_7/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*,
shared_nameSGD/dense_7/kernel/momentum
�
/SGD/dense_7/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_7/kernel/momentum*
_output_shapes

:@*
dtype0
�
'SGD/batch_normalization_9/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'SGD/batch_normalization_9/beta/momentum
�
;SGD/batch_normalization_9/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_9/beta/momentum*
_output_shapes
:@*
dtype0
�
(SGD/batch_normalization_9/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(SGD/batch_normalization_9/gamma/momentum
�
<SGD/batch_normalization_9/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_9/gamma/momentum*
_output_shapes
:@*
dtype0
�
SGD/dense_6/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameSGD/dense_6/bias/momentum
�
-SGD/dense_6/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_6/bias/momentum*
_output_shapes
:@*
dtype0
�
SGD/dense_6/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*,
shared_nameSGD/dense_6/kernel/momentum
�
/SGD/dense_6/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_6/kernel/momentum*
_output_shapes
:	�@*
dtype0
�
'SGD/batch_normalization_8/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'SGD/batch_normalization_8/beta/momentum
�
;SGD/batch_normalization_8/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_8/beta/momentum*
_output_shapes
:@*
dtype0
�
(SGD/batch_normalization_8/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(SGD/batch_normalization_8/gamma/momentum
�
<SGD/batch_normalization_8/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_8/gamma/momentum*
_output_shapes
:@*
dtype0
�
SGD/conv2d_6/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/conv2d_6/bias/momentum
�
.SGD/conv2d_6/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_6/bias/momentum*
_output_shapes
:@*
dtype0
�
SGD/conv2d_6/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*-
shared_nameSGD/conv2d_6/kernel/momentum
�
0SGD/conv2d_6/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_6/kernel/momentum*&
_output_shapes
: @*
dtype0
�
'SGD/batch_normalization_7/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'SGD/batch_normalization_7/beta/momentum
�
;SGD/batch_normalization_7/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_7/beta/momentum*
_output_shapes
: *
dtype0
�
(SGD/batch_normalization_7/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_7/gamma/momentum
�
<SGD/batch_normalization_7/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_7/gamma/momentum*
_output_shapes
: *
dtype0
�
SGD/conv2d_5/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/conv2d_5/bias/momentum
�
.SGD/conv2d_5/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_5/bias/momentum*
_output_shapes
: *
dtype0
�
SGD/conv2d_5/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameSGD/conv2d_5/kernel/momentum
�
0SGD/conv2d_5/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_5/kernel/momentum*&
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:@*
dtype0
�
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_9/moving_variance
�
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
:@*
dtype0
�
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_9/moving_mean
�
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
:@*
dtype0
�
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_9/beta
�
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_9/gamma
�
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
:@*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:@*
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	�@*
dtype0
�
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_8/moving_variance
�
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:@*
dtype0
�
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_8/moving_mean
�
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:@*
dtype0
�
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_8/beta
�
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_8/gamma
�
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:@*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0
�
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: @*
dtype0
�
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_7/moving_variance
�
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
: *
dtype0
�
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_7/moving_mean
�
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
: *
dtype0
�
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_7/beta
�
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
: *
dtype0
�
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_7/gamma
�
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
: *
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
: *
dtype0
�
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
: *
dtype0
�
serving_default_conv2d_5_inputPlaceholder*/
_output_shapes
:���������>*
dtype0*$
shape:���������>
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_5_inputconv2d_5/kernelconv2d_5/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancedense_6/kerneldense_6/bias%batch_normalization_9/moving_variancebatch_normalization_9/gamma!batch_normalization_9/moving_meanbatch_normalization_9/betadense_7/kerneldense_7/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_1456835

NoOpNoOp
�i
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�i
value�iB�i B�i
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$axis
	%gamma
&beta
'moving_mean
(moving_variance*
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses* 
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias
 7_jit_compiled_convolution_op*
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance*
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses* 
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses* 
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
U_random_generator* 
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias*
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance*
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

okernel
pbias*
�
0
1
%2
&3
'4
(5
56
67
?8
@9
A10
B11
\12
]13
e14
f15
g16
h17
o18
p19*
j
0
1
%2
&3
54
65
?6
@7
\8
]9
e10
f11
o12
p13*

q0
r1
s2* 
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ytrace_0
ztrace_1
{trace_2
|trace_3* 
7
}trace_0
~trace_1
trace_2
�trace_3* 
* 
�
	�iter

�decay
�learning_rate
�momentummomentum�momentum�%momentum�&momentum�5momentum�6momentum�?momentum�@momentum�\momentum�]momentum�emomentum�fmomentum�omomentum�pmomentum�*

�serving_default* 

0
1*

0
1*
	
q0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
%0
&1
'2
(3*

%0
&1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

50
61*

50
61*
	
r0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
?0
@1
A2
B3*

?0
@1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

\0
]1*

\0
]1*
	
s0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
e0
f1
g2
h3*

e0
f1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

o0
p1*

o0
p1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 
.
'0
(1
A2
B3
g4
h5*
R
0
1
2
3
4
5
6
7
	8

9
10*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
KE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
	
q0* 
* 
* 
* 

'0
(1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
r0* 
* 
* 
* 

A0
B1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
s0* 
* 
* 
* 

g0
h1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
��
VARIABLE_VALUESGD/conv2d_5/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/conv2d_5/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE(SGD/batch_normalization_7/gamma/momentumXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE'SGD/batch_normalization_7/beta/momentumWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/conv2d_6/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/conv2d_6/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE(SGD/batch_normalization_8/gamma/momentumXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE'SGD/batch_normalization_8/beta/momentumWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense_6/kernel/momentumYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense_6/bias/momentumWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE(SGD/batch_normalization_9/gamma/momentumXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE'SGD/batch_normalization_9/beta/momentumWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense_7/kernel/momentumYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense_7/bias/momentumWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_5/kernelconv2d_5/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancedense_6/kerneldense_6/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_variancedense_7/kerneldense_7/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotal_1count_1totalcountSGD/conv2d_5/kernel/momentumSGD/conv2d_5/bias/momentum(SGD/batch_normalization_7/gamma/momentum'SGD/batch_normalization_7/beta/momentumSGD/conv2d_6/kernel/momentumSGD/conv2d_6/bias/momentum(SGD/batch_normalization_8/gamma/momentum'SGD/batch_normalization_8/beta/momentumSGD/dense_6/kernel/momentumSGD/dense_6/bias/momentum(SGD/batch_normalization_9/gamma/momentum'SGD/batch_normalization_9/beta/momentumSGD/dense_7/kernel/momentumSGD/dense_7/bias/momentumConst*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_1457935
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_5/kernelconv2d_5/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancedense_6/kerneldense_6/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_variancedense_7/kerneldense_7/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotal_1count_1totalcountSGD/conv2d_5/kernel/momentumSGD/conv2d_5/bias/momentum(SGD/batch_normalization_7/gamma/momentum'SGD/batch_normalization_7/beta/momentumSGD/conv2d_6/kernel/momentumSGD/conv2d_6/bias/momentum(SGD/batch_normalization_8/gamma/momentum'SGD/batch_normalization_8/beta/momentumSGD/dense_6/kernel/momentumSGD/dense_6/bias/momentum(SGD/batch_normalization_9/gamma/momentum'SGD/batch_normalization_9/beta/momentumSGD/dense_7/kernel/momentumSGD/dense_7/bias/momentum*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_1458071��
�
h
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1457435

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_5_layer_call_fn_1457430

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1455930�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_1457624Q
7conv2d_5_kernel_regularizer_abs_readvariableop_resource: 
identity��.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOpf
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_5_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp7conv2d_5_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_5/kernel/Regularizer/L2LossL2Loss9conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0+conv2d_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_5/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp
�%
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1457566

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1455821

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
.__inference_sequential_3_layer_call_fn_1456964

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:	�@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_3_layer_call_and_return_conditional_losses_1456555o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������>: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������>
 
_user_specified_nameinputs
�
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_1457473

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�'
 __inference__traced_save_1457935
file_prefix@
&read_disablecopyonread_conv2d_5_kernel: 4
&read_1_disablecopyonread_conv2d_5_bias: B
4read_2_disablecopyonread_batch_normalization_7_gamma: A
3read_3_disablecopyonread_batch_normalization_7_beta: H
:read_4_disablecopyonread_batch_normalization_7_moving_mean: L
>read_5_disablecopyonread_batch_normalization_7_moving_variance: B
(read_6_disablecopyonread_conv2d_6_kernel: @4
&read_7_disablecopyonread_conv2d_6_bias:@B
4read_8_disablecopyonread_batch_normalization_8_gamma:@A
3read_9_disablecopyonread_batch_normalization_8_beta:@I
;read_10_disablecopyonread_batch_normalization_8_moving_mean:@M
?read_11_disablecopyonread_batch_normalization_8_moving_variance:@;
(read_12_disablecopyonread_dense_6_kernel:	�@4
&read_13_disablecopyonread_dense_6_bias:@C
5read_14_disablecopyonread_batch_normalization_9_gamma:@B
4read_15_disablecopyonread_batch_normalization_9_beta:@I
;read_16_disablecopyonread_batch_normalization_9_moving_mean:@M
?read_17_disablecopyonread_batch_normalization_9_moving_variance:@:
(read_18_disablecopyonread_dense_7_kernel:@4
&read_19_disablecopyonread_dense_7_bias:,
"read_20_disablecopyonread_sgd_iter:	 -
#read_21_disablecopyonread_sgd_decay: 5
+read_22_disablecopyonread_sgd_learning_rate: 0
&read_23_disablecopyonread_sgd_momentum: +
!read_24_disablecopyonread_total_1: +
!read_25_disablecopyonread_count_1: )
read_26_disablecopyonread_total: )
read_27_disablecopyonread_count: P
6read_28_disablecopyonread_sgd_conv2d_5_kernel_momentum: B
4read_29_disablecopyonread_sgd_conv2d_5_bias_momentum: P
Bread_30_disablecopyonread_sgd_batch_normalization_7_gamma_momentum: O
Aread_31_disablecopyonread_sgd_batch_normalization_7_beta_momentum: P
6read_32_disablecopyonread_sgd_conv2d_6_kernel_momentum: @B
4read_33_disablecopyonread_sgd_conv2d_6_bias_momentum:@P
Bread_34_disablecopyonread_sgd_batch_normalization_8_gamma_momentum:@O
Aread_35_disablecopyonread_sgd_batch_normalization_8_beta_momentum:@H
5read_36_disablecopyonread_sgd_dense_6_kernel_momentum:	�@A
3read_37_disablecopyonread_sgd_dense_6_bias_momentum:@P
Bread_38_disablecopyonread_sgd_batch_normalization_9_gamma_momentum:@O
Aread_39_disablecopyonread_sgd_batch_normalization_9_beta_momentum:@G
5read_40_disablecopyonread_sgd_dense_7_kernel_momentum:@A
3read_41_disablecopyonread_sgd_dense_7_bias_momentum:
savev2_const
identity_85��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_conv2d_5_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
: z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_conv2d_5_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_conv2d_5_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_2/DisableCopyOnReadDisableCopyOnRead4read_2_disablecopyonread_batch_normalization_7_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp4read_2_disablecopyonread_batch_normalization_7_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_3/DisableCopyOnReadDisableCopyOnRead3read_3_disablecopyonread_batch_normalization_7_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp3read_3_disablecopyonread_batch_normalization_7_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_4/DisableCopyOnReadDisableCopyOnRead:read_4_disablecopyonread_batch_normalization_7_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp:read_4_disablecopyonread_batch_normalization_7_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_5/DisableCopyOnReadDisableCopyOnRead>read_5_disablecopyonread_batch_normalization_7_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp>read_5_disablecopyonread_batch_normalization_7_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_conv2d_6_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
: @z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_conv2d_6_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_conv2d_6_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_8/DisableCopyOnReadDisableCopyOnRead4read_8_disablecopyonread_batch_normalization_8_gamma"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp4read_8_disablecopyonread_batch_normalization_8_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_9/DisableCopyOnReadDisableCopyOnRead3read_9_disablecopyonread_batch_normalization_8_beta"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp3read_9_disablecopyonread_batch_normalization_8_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_10/DisableCopyOnReadDisableCopyOnRead;read_10_disablecopyonread_batch_normalization_8_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp;read_10_disablecopyonread_batch_normalization_8_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_11/DisableCopyOnReadDisableCopyOnRead?read_11_disablecopyonread_batch_normalization_8_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp?read_11_disablecopyonread_batch_normalization_8_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_dense_6_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@{
Read_13/DisableCopyOnReadDisableCopyOnRead&read_13_disablecopyonread_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp&read_13_disablecopyonread_dense_6_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_14/DisableCopyOnReadDisableCopyOnRead5read_14_disablecopyonread_batch_normalization_9_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp5read_14_disablecopyonread_batch_normalization_9_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_15/DisableCopyOnReadDisableCopyOnRead4read_15_disablecopyonread_batch_normalization_9_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp4read_15_disablecopyonread_batch_normalization_9_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_16/DisableCopyOnReadDisableCopyOnRead;read_16_disablecopyonread_batch_normalization_9_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp;read_16_disablecopyonread_batch_normalization_9_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_17/DisableCopyOnReadDisableCopyOnRead?read_17_disablecopyonread_batch_normalization_9_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp?read_17_disablecopyonread_batch_normalization_9_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_18/DisableCopyOnReadDisableCopyOnRead(read_18_disablecopyonread_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp(read_18_disablecopyonread_dense_7_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

:@{
Read_19/DisableCopyOnReadDisableCopyOnRead&read_19_disablecopyonread_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp&read_19_disablecopyonread_dense_7_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:w
Read_20/DisableCopyOnReadDisableCopyOnRead"read_20_disablecopyonread_sgd_iter"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp"read_20_disablecopyonread_sgd_iter^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0	*
_output_shapes
: x
Read_21/DisableCopyOnReadDisableCopyOnRead#read_21_disablecopyonread_sgd_decay"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp#read_21_disablecopyonread_sgd_decay^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_22/DisableCopyOnReadDisableCopyOnRead+read_22_disablecopyonread_sgd_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp+read_22_disablecopyonread_sgd_learning_rate^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_23/DisableCopyOnReadDisableCopyOnRead&read_23_disablecopyonread_sgd_momentum"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp&read_23_disablecopyonread_sgd_momentum^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_24/DisableCopyOnReadDisableCopyOnRead!read_24_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp!read_24_disablecopyonread_total_1^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_25/DisableCopyOnReadDisableCopyOnRead!read_25_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp!read_25_disablecopyonread_count_1^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_26/DisableCopyOnReadDisableCopyOnReadread_26_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOpread_26_disablecopyonread_total^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_27/DisableCopyOnReadDisableCopyOnReadread_27_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOpread_27_disablecopyonread_count^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_28/DisableCopyOnReadDisableCopyOnRead6read_28_disablecopyonread_sgd_conv2d_5_kernel_momentum"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp6read_28_disablecopyonread_sgd_conv2d_5_kernel_momentum^Read_28/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_29/DisableCopyOnReadDisableCopyOnRead4read_29_disablecopyonread_sgd_conv2d_5_bias_momentum"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp4read_29_disablecopyonread_sgd_conv2d_5_bias_momentum^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_30/DisableCopyOnReadDisableCopyOnReadBread_30_disablecopyonread_sgd_batch_normalization_7_gamma_momentum"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOpBread_30_disablecopyonread_sgd_batch_normalization_7_gamma_momentum^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_31/DisableCopyOnReadDisableCopyOnReadAread_31_disablecopyonread_sgd_batch_normalization_7_beta_momentum"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOpAread_31_disablecopyonread_sgd_batch_normalization_7_beta_momentum^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_32/DisableCopyOnReadDisableCopyOnRead6read_32_disablecopyonread_sgd_conv2d_6_kernel_momentum"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp6read_32_disablecopyonread_sgd_conv2d_6_kernel_momentum^Read_32/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_33/DisableCopyOnReadDisableCopyOnRead4read_33_disablecopyonread_sgd_conv2d_6_bias_momentum"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp4read_33_disablecopyonread_sgd_conv2d_6_bias_momentum^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_34/DisableCopyOnReadDisableCopyOnReadBread_34_disablecopyonread_sgd_batch_normalization_8_gamma_momentum"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOpBread_34_disablecopyonread_sgd_batch_normalization_8_gamma_momentum^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_35/DisableCopyOnReadDisableCopyOnReadAread_35_disablecopyonread_sgd_batch_normalization_8_beta_momentum"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOpAread_35_disablecopyonread_sgd_batch_normalization_8_beta_momentum^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_36/DisableCopyOnReadDisableCopyOnRead5read_36_disablecopyonread_sgd_dense_6_kernel_momentum"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp5read_36_disablecopyonread_sgd_dense_6_kernel_momentum^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_37/DisableCopyOnReadDisableCopyOnRead3read_37_disablecopyonread_sgd_dense_6_bias_momentum"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp3read_37_disablecopyonread_sgd_dense_6_bias_momentum^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_38/DisableCopyOnReadDisableCopyOnReadBread_38_disablecopyonread_sgd_batch_normalization_9_gamma_momentum"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOpBread_38_disablecopyonread_sgd_batch_normalization_9_gamma_momentum^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_39/DisableCopyOnReadDisableCopyOnReadAread_39_disablecopyonread_sgd_batch_normalization_9_beta_momentum"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOpAread_39_disablecopyonread_sgd_batch_normalization_9_beta_momentum^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_40/DisableCopyOnReadDisableCopyOnRead5read_40_disablecopyonread_sgd_dense_7_kernel_momentum"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp5read_40_disablecopyonread_sgd_dense_7_kernel_momentum^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_41/DisableCopyOnReadDisableCopyOnRead3read_41_disablecopyonread_sgd_dense_7_bias_momentum"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp3read_41_disablecopyonread_sgd_dense_7_bias_momentum^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *9
dtypes/
-2+	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_84Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_85IdentityIdentity_84:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_85Identity_85:output:0*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:+

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
h
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1455854

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1457330

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�k
�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1456555

inputs*
conv2d_5_1456464: 
conv2d_5_1456466: +
batch_normalization_7_1456469: +
batch_normalization_7_1456471: +
batch_normalization_7_1456473: +
batch_normalization_7_1456475: *
conv2d_6_1456479: @
conv2d_6_1456481:@+
batch_normalization_8_1456484:@+
batch_normalization_8_1456486:@+
batch_normalization_8_1456488:@+
batch_normalization_8_1456490:@"
dense_6_1456496:	�@
dense_6_1456498:@+
batch_normalization_9_1456501:@+
batch_normalization_9_1456503:@+
batch_normalization_9_1456505:@+
batch_normalization_9_1456507:@!
dense_7_1456510:@
dense_7_1456512:
identity��-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp� conv2d_6/StatefulPartitionedCall�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense_6/StatefulPartitionedCall�-dense_6/kernel/Regularizer/Abs/ReadVariableOp�0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense_7/StatefulPartitionedCall�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_1456464conv2d_5_1456466*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������> *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1456046�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_7_1456469batch_normalization_7_1456471batch_normalization_7_1456473batch_normalization_7_1456475*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������> *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1455821�
max_pooling2d_4/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1455854�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_6_1456479conv2d_6_1456481*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1456086�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_8_1456484batch_normalization_8_1456486batch_normalization_8_1456488batch_normalization_8_1456490*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1455897�
max_pooling2d_5/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1455930�
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1456108�
dropout_3/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1456258�
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_6_1456496dense_6_1456498*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1456148�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_9_1456501batch_normalization_9_1456503batch_normalization_9_1456505batch_normalization_9_1456507*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1455991�
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0dense_7_1456510dense_7_1456512*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1456174f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_5_1456464*&
_output_shapes
: *
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_5_1456464*&
_output_shapes
: *
dtype0�
"conv2d_5/kernel/Regularizer/L2LossL2Loss9conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0+conv2d_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_6_1456479*&
_output_shapes
: @*
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_6_1456479*&
_output_shapes
: @*
dtype0�
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
-dense_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_6_1456496*
_output_shapes
:	�@*
dtype0�
dense_6/kernel/Regularizer/AbsAbs5dense_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@s
"dense_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_6/kernel/Regularizer/SumSum"dense_6/kernel/Regularizer/Abs:y:0+dense_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_6/kernel/Regularizer/addAddV2)dense_6/kernel/Regularizer/Const:output:0"dense_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_1456496*
_output_shapes
:	�@*
dtype0�
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_6/kernel/Regularizer/mul_1Mul+dense_6/kernel/Regularizer/mul_1/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
 dense_6/kernel/Regularizer/add_1AddV2"dense_6/kernel/Regularizer/add:z:0$dense_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_6/StatefulPartitionedCall.^dense_6/kernel/Regularizer/Abs/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������>: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2^
-dense_6/kernel/Regularizer/Abs/ReadVariableOp-dense_6/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:W S
/
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1455991

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
.__inference_sequential_3_layer_call_fn_1456459
conv2d_5_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:	�@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_3_layer_call_and_return_conditional_losses_1456416o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������>: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:���������>
(
_user_specified_nameconv2d_5_input
�
�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1457407

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1457363

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
*__inference_conv2d_6_layer_call_fn_1457339

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1456086w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�l
�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1456220
conv2d_5_input*
conv2d_5_1456047: 
conv2d_5_1456049: +
batch_normalization_7_1456052: +
batch_normalization_7_1456054: +
batch_normalization_7_1456056: +
batch_normalization_7_1456058: *
conv2d_6_1456087: @
conv2d_6_1456089:@+
batch_normalization_8_1456092:@+
batch_normalization_8_1456094:@+
batch_normalization_8_1456096:@+
batch_normalization_8_1456098:@"
dense_6_1456149:	�@
dense_6_1456151:@+
batch_normalization_9_1456154:@+
batch_normalization_9_1456156:@+
batch_normalization_9_1456158:@+
batch_normalization_9_1456160:@!
dense_7_1456175:@
dense_7_1456177:
identity��-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp� conv2d_6/StatefulPartitionedCall�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense_6/StatefulPartitionedCall�-dense_6/kernel/Regularizer/Abs/ReadVariableOp�0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense_7/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_1456047conv2d_5_1456049*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������> *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1456046�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_7_1456052batch_normalization_7_1456054batch_normalization_7_1456056batch_normalization_7_1456058*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������> *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1455803�
max_pooling2d_4/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1455854�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_6_1456087conv2d_6_1456089*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1456086�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_8_1456092batch_normalization_8_1456094batch_normalization_8_1456096batch_normalization_8_1456098*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1455879�
max_pooling2d_5/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1455930�
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1456108�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1456122�
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_6_1456149dense_6_1456151*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1456148�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_9_1456154batch_normalization_9_1456156batch_normalization_9_1456158batch_normalization_9_1456160*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1455971�
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0dense_7_1456175dense_7_1456177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1456174f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_5_1456047*&
_output_shapes
: *
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_5_1456047*&
_output_shapes
: *
dtype0�
"conv2d_5/kernel/Regularizer/L2LossL2Loss9conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0+conv2d_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_6_1456087*&
_output_shapes
: @*
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_6_1456087*&
_output_shapes
: @*
dtype0�
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
-dense_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_6_1456149*
_output_shapes
:	�@*
dtype0�
dense_6/kernel/Regularizer/AbsAbs5dense_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@s
"dense_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_6/kernel/Regularizer/SumSum"dense_6/kernel/Regularizer/Abs:y:0+dense_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_6/kernel/Regularizer/addAddV2)dense_6/kernel/Regularizer/Const:output:0"dense_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_1456149*
_output_shapes
:	�@*
dtype0�
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_6/kernel/Regularizer/mul_1Mul+dense_6/kernel/Regularizer/mul_1/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
 dense_6/kernel/Regularizer/add_1AddV2"dense_6/kernel/Regularizer/add:z:0$dense_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_6/StatefulPartitionedCall.^dense_6/kernel/Regularizer/Abs/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_7/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������>: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2^
-dense_6/kernel/Regularizer/Abs/ReadVariableOp-dense_6/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:_ [
/
_output_shapes
:���������>
(
_user_specified_nameconv2d_5_input
�

e
F__inference_dropout_3_layer_call_and_return_conditional_losses_1457468

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_dense_6_layer_call_fn_1457482

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1456148o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_dense_6_layer_call_and_return_conditional_losses_1457506

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�-dense_6/kernel/Regularizer/Abs/ReadVariableOp�0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@e
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
-dense_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_6/kernel/Regularizer/AbsAbs5dense_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@s
"dense_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_6/kernel/Regularizer/SumSum"dense_6/kernel/Regularizer/Abs:y:0+dense_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_6/kernel/Regularizer/addAddV2)dense_6/kernel/Regularizer/Const:output:0"dense_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_6/kernel/Regularizer/mul_1Mul+dense_6/kernel/Regularizer/mul_1/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
 dense_6/kernel/Regularizer/add_1AddV2"dense_6/kernel/Regularizer/add:z:0$dense_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_6/kernel/Regularizer/Abs/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_6/kernel/Regularizer/Abs/ReadVariableOp-dense_6/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_dense_7_layer_call_and_return_conditional_losses_1456174

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
D__inference_dense_7_layer_call_and_return_conditional_losses_1457606

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
*__inference_conv2d_5_layer_call_fn_1457234

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������> *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1456046w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������> `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������>: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������>
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_4_layer_call_fn_1457325

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1455854�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
G
+__inference_dropout_3_layer_call_fn_1457456

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1456258a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1457425

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1455897

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_1456835
conv2d_5_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:	�@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_1455784o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������>: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:���������>
(
_user_specified_nameconv2d_5_input
�	
�
7__inference_batch_normalization_8_layer_call_fn_1457376

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1455879�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1456046

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������> *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������> X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������> f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_5/kernel/Regularizer/L2LossL2Loss9conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0+conv2d_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������> �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
D__inference_dense_6_layer_call_and_return_conditional_losses_1456148

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�-dense_6/kernel/Regularizer/Abs/ReadVariableOp�0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@e
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
-dense_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_6/kernel/Regularizer/AbsAbs5dense_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@s
"dense_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_6/kernel/Regularizer/SumSum"dense_6/kernel/Regularizer/Abs:y:0+dense_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_6/kernel/Regularizer/addAddV2)dense_6/kernel/Regularizer/Const:output:0"dense_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_6/kernel/Regularizer/mul_1Mul+dense_6/kernel/Regularizer/mul_1/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
 dense_6/kernel/Regularizer/add_1AddV2"dense_6/kernel/Regularizer/add:z:0$dense_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_6/kernel/Regularizer/Abs/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_6/kernel/Regularizer/Abs/ReadVariableOp-dense_6/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_1457642Q
7conv2d_6_kernel_regularizer_abs_readvariableop_resource: @
identity��.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpf
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_6_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp7conv2d_6_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: @*
dtype0�
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_6/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp
�

e
F__inference_dropout_3_layer_call_and_return_conditional_losses_1456122

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�k
�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1456319
conv2d_5_input*
conv2d_5_1456223: 
conv2d_5_1456225: +
batch_normalization_7_1456228: +
batch_normalization_7_1456230: +
batch_normalization_7_1456232: +
batch_normalization_7_1456234: *
conv2d_6_1456238: @
conv2d_6_1456240:@+
batch_normalization_8_1456243:@+
batch_normalization_8_1456245:@+
batch_normalization_8_1456247:@+
batch_normalization_8_1456249:@"
dense_6_1456260:	�@
dense_6_1456262:@+
batch_normalization_9_1456265:@+
batch_normalization_9_1456267:@+
batch_normalization_9_1456269:@+
batch_normalization_9_1456271:@!
dense_7_1456274:@
dense_7_1456276:
identity��-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp� conv2d_6/StatefulPartitionedCall�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense_6/StatefulPartitionedCall�-dense_6/kernel/Regularizer/Abs/ReadVariableOp�0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense_7/StatefulPartitionedCall�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_1456223conv2d_5_1456225*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������> *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1456046�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_7_1456228batch_normalization_7_1456230batch_normalization_7_1456232batch_normalization_7_1456234*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������> *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1455821�
max_pooling2d_4/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1455854�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_6_1456238conv2d_6_1456240*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1456086�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_8_1456243batch_normalization_8_1456245batch_normalization_8_1456247batch_normalization_8_1456249*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1455897�
max_pooling2d_5/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1455930�
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1456108�
dropout_3/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1456258�
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_6_1456260dense_6_1456262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1456148�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_9_1456265batch_normalization_9_1456267batch_normalization_9_1456269batch_normalization_9_1456271*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1455991�
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0dense_7_1456274dense_7_1456276*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1456174f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_5_1456223*&
_output_shapes
: *
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_5_1456223*&
_output_shapes
: *
dtype0�
"conv2d_5/kernel/Regularizer/L2LossL2Loss9conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0+conv2d_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_6_1456238*&
_output_shapes
: @*
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_6_1456238*&
_output_shapes
: @*
dtype0�
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
-dense_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_6_1456260*
_output_shapes
:	�@*
dtype0�
dense_6/kernel/Regularizer/AbsAbs5dense_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@s
"dense_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_6/kernel/Regularizer/SumSum"dense_6/kernel/Regularizer/Abs:y:0+dense_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_6/kernel/Regularizer/addAddV2)dense_6/kernel/Regularizer/Const:output:0"dense_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_1456260*
_output_shapes
:	�@*
dtype0�
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_6/kernel/Regularizer/mul_1Mul+dense_6/kernel/Regularizer/mul_1/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
 dense_6/kernel/Regularizer/add_1AddV2"dense_6/kernel/Regularizer/add:z:0$dense_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_6/StatefulPartitionedCall.^dense_6/kernel/Regularizer/Abs/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������>: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2^
-dense_6/kernel/Regularizer/Abs/ReadVariableOp-dense_6/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:_ [
/
_output_shapes
:���������>
(
_user_specified_nameconv2d_5_input
�	
�
7__inference_batch_normalization_7_layer_call_fn_1457271

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1455803�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_9_layer_call_fn_1457532

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1455991o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_8_layer_call_fn_1457389

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1455897�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_1456258

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_dense_7_layer_call_fn_1457595

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1456174o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1455879

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�%
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1455971

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_7_layer_call_fn_1457284

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1455821�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1455803

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_1457660I
6dense_6_kernel_regularizer_abs_readvariableop_resource:	�@
identity��-dense_6/kernel/Regularizer/Abs/ReadVariableOp�0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpe
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
-dense_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_6_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_6/kernel/Regularizer/AbsAbs5dense_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@s
"dense_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_6/kernel/Regularizer/SumSum"dense_6/kernel/Regularizer/Abs:y:0+dense_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_6/kernel/Regularizer/addAddV2)dense_6/kernel/Regularizer/Const:output:0"dense_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp6dense_6_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_6/kernel/Regularizer/mul_1Mul+dense_6/kernel/Regularizer/mul_1/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
 dense_6/kernel/Regularizer/add_1AddV2"dense_6/kernel/Regularizer/add:z:0$dense_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_6/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp.^dense_6/kernel/Regularizer/Abs/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-dense_6/kernel/Regularizer/Abs/ReadVariableOp-dense_6/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
.__inference_sequential_3_layer_call_fn_1456919

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:	�@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_3_layer_call_and_return_conditional_losses_1456416o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������>: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1457586

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1457258

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������> *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������> X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������> f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_5/kernel/Regularizer/L2LossL2Loss9conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0+conv2d_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������> �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������>
 
_user_specified_nameinputs
�
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_1457446

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
.__inference_sequential_3_layer_call_fn_1456598
conv2d_5_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:	�@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_3_layer_call_and_return_conditional_losses_1456555o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������>: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:���������>
(
_user_specified_nameconv2d_5_input
�{
�
"__inference__wrapped_model_1455784
conv2d_5_inputN
4sequential_3_conv2d_5_conv2d_readvariableop_resource: C
5sequential_3_conv2d_5_biasadd_readvariableop_resource: H
:sequential_3_batch_normalization_7_readvariableop_resource: J
<sequential_3_batch_normalization_7_readvariableop_1_resource: Y
Ksequential_3_batch_normalization_7_fusedbatchnormv3_readvariableop_resource: [
Msequential_3_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: N
4sequential_3_conv2d_6_conv2d_readvariableop_resource: @C
5sequential_3_conv2d_6_biasadd_readvariableop_resource:@H
:sequential_3_batch_normalization_8_readvariableop_resource:@J
<sequential_3_batch_normalization_8_readvariableop_1_resource:@Y
Ksequential_3_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@[
Msequential_3_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@F
3sequential_3_dense_6_matmul_readvariableop_resource:	�@B
4sequential_3_dense_6_biasadd_readvariableop_resource:@R
Dsequential_3_batch_normalization_9_batchnorm_readvariableop_resource:@V
Hsequential_3_batch_normalization_9_batchnorm_mul_readvariableop_resource:@T
Fsequential_3_batch_normalization_9_batchnorm_readvariableop_1_resource:@T
Fsequential_3_batch_normalization_9_batchnorm_readvariableop_2_resource:@E
3sequential_3_dense_7_matmul_readvariableop_resource:@B
4sequential_3_dense_7_biasadd_readvariableop_resource:
identity��Bsequential_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp�Dsequential_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�1sequential_3/batch_normalization_7/ReadVariableOp�3sequential_3/batch_normalization_7/ReadVariableOp_1�Bsequential_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�Dsequential_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�1sequential_3/batch_normalization_8/ReadVariableOp�3sequential_3/batch_normalization_8/ReadVariableOp_1�;sequential_3/batch_normalization_9/batchnorm/ReadVariableOp�=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1�=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2�?sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp�,sequential_3/conv2d_5/BiasAdd/ReadVariableOp�+sequential_3/conv2d_5/Conv2D/ReadVariableOp�,sequential_3/conv2d_6/BiasAdd/ReadVariableOp�+sequential_3/conv2d_6/Conv2D/ReadVariableOp�+sequential_3/dense_6/BiasAdd/ReadVariableOp�*sequential_3/dense_6/MatMul/ReadVariableOp�+sequential_3/dense_7/BiasAdd/ReadVariableOp�*sequential_3/dense_7/MatMul/ReadVariableOp�
+sequential_3/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
sequential_3/conv2d_5/Conv2DConv2Dconv2d_5_input3sequential_3/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������> *
paddingSAME*
strides
�
,sequential_3/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_3/conv2d_5/BiasAddBiasAdd%sequential_3/conv2d_5/Conv2D:output:04sequential_3/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������> �
sequential_3/conv2d_5/ReluRelu&sequential_3/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:���������> �
1sequential_3/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0�
3sequential_3/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Bsequential_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
Dsequential_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
3sequential_3/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3(sequential_3/conv2d_5/Relu:activations:09sequential_3/batch_normalization_7/ReadVariableOp:value:0;sequential_3/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������> : : : : :*
epsilon%o�:*
is_training( �
$sequential_3/max_pooling2d_4/MaxPoolMaxPool7sequential_3/batch_normalization_7/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
�
+sequential_3/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
sequential_3/conv2d_6/Conv2DConv2D-sequential_3/max_pooling2d_4/MaxPool:output:03sequential_3/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
,sequential_3/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_3/conv2d_6/BiasAddBiasAdd%sequential_3/conv2d_6/Conv2D:output:04sequential_3/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
sequential_3/conv2d_6/ReluRelu&sequential_3/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
1sequential_3/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype0�
3sequential_3/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Bsequential_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Dsequential_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3sequential_3/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3(sequential_3/conv2d_6/Relu:activations:09sequential_3/batch_normalization_8/ReadVariableOp:value:0;sequential_3/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
$sequential_3/max_pooling2d_5/MaxPoolMaxPool7sequential_3/batch_normalization_8/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
m
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  �
sequential_3/flatten_3/ReshapeReshape-sequential_3/max_pooling2d_5/MaxPool:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:�����������
sequential_3/dropout_3/IdentityIdentity'sequential_3/flatten_3/Reshape:output:0*
T0*(
_output_shapes
:�����������
*sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
sequential_3/dense_6/MatMulMatMul(sequential_3/dropout_3/Identity:output:02sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_3/dense_6/BiasAddBiasAdd%sequential_3/dense_6/MatMul:product:03sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
sequential_3/dense_6/ReluRelu%sequential_3/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
;sequential_3/batch_normalization_9/batchnorm/ReadVariableOpReadVariableOpDsequential_3_batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0w
2sequential_3/batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
0sequential_3/batch_normalization_9/batchnorm/addAddV2Csequential_3/batch_normalization_9/batchnorm/ReadVariableOp:value:0;sequential_3/batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
2sequential_3/batch_normalization_9/batchnorm/RsqrtRsqrt4sequential_3/batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:@�
?sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_3_batch_normalization_9_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
0sequential_3/batch_normalization_9/batchnorm/mulMul6sequential_3/batch_normalization_9/batchnorm/Rsqrt:y:0Gsequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
2sequential_3/batch_normalization_9/batchnorm/mul_1Mul'sequential_3/dense_6/Relu:activations:04sequential_3/batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_3_batch_normalization_9_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
2sequential_3/batch_normalization_9/batchnorm/mul_2MulEsequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1:value:04sequential_3/batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_3_batch_normalization_9_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
0sequential_3/batch_normalization_9/batchnorm/subSubEsequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2:value:06sequential_3/batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
2sequential_3/batch_normalization_9/batchnorm/add_1AddV26sequential_3/batch_normalization_9/batchnorm/mul_1:z:04sequential_3/batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
*sequential_3/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_7_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
sequential_3/dense_7/MatMulMatMul6sequential_3/batch_normalization_9/batchnorm/add_1:z:02sequential_3/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+sequential_3/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_3/dense_7/BiasAddBiasAdd%sequential_3/dense_7/MatMul:product:03sequential_3/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
sequential_3/dense_7/SigmoidSigmoid%sequential_3/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������o
IdentityIdentity sequential_3/dense_7/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOpC^sequential_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_7/ReadVariableOp4^sequential_3/batch_normalization_7/ReadVariableOp_1C^sequential_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_3/batch_normalization_8/ReadVariableOp4^sequential_3/batch_normalization_8/ReadVariableOp_1<^sequential_3/batch_normalization_9/batchnorm/ReadVariableOp>^sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1>^sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2@^sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp-^sequential_3/conv2d_5/BiasAdd/ReadVariableOp,^sequential_3/conv2d_5/Conv2D/ReadVariableOp-^sequential_3/conv2d_6/BiasAdd/ReadVariableOp,^sequential_3/conv2d_6/Conv2D/ReadVariableOp,^sequential_3/dense_6/BiasAdd/ReadVariableOp+^sequential_3/dense_6/MatMul/ReadVariableOp,^sequential_3/dense_7/BiasAdd/ReadVariableOp+^sequential_3/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������>: : : : : : : : : : : : : : : : : : : : 2�
Dsequential_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12�
Bsequential_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2j
3sequential_3/batch_normalization_7/ReadVariableOp_13sequential_3/batch_normalization_7/ReadVariableOp_12f
1sequential_3/batch_normalization_7/ReadVariableOp1sequential_3/batch_normalization_7/ReadVariableOp2�
Dsequential_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12�
Bsequential_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2j
3sequential_3/batch_normalization_8/ReadVariableOp_13sequential_3/batch_normalization_8/ReadVariableOp_12f
1sequential_3/batch_normalization_8/ReadVariableOp1sequential_3/batch_normalization_8/ReadVariableOp2~
=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_12~
=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_22z
;sequential_3/batch_normalization_9/batchnorm/ReadVariableOp;sequential_3/batch_normalization_9/batchnorm/ReadVariableOp2�
?sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp?sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp2\
,sequential_3/conv2d_5/BiasAdd/ReadVariableOp,sequential_3/conv2d_5/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_5/Conv2D/ReadVariableOp+sequential_3/conv2d_5/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_6/BiasAdd/ReadVariableOp,sequential_3/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_6/Conv2D/ReadVariableOp+sequential_3/conv2d_6/Conv2D/ReadVariableOp2Z
+sequential_3/dense_6/BiasAdd/ReadVariableOp+sequential_3/dense_6/BiasAdd/ReadVariableOp2X
*sequential_3/dense_6/MatMul/ReadVariableOp*sequential_3/dense_6/MatMul/ReadVariableOp2Z
+sequential_3/dense_7/BiasAdd/ReadVariableOp+sequential_3/dense_7/BiasAdd/ReadVariableOp2X
*sequential_3/dense_7/MatMul/ReadVariableOp*sequential_3/dense_7/MatMul/ReadVariableOp:_ [
/
_output_shapes
:���������>
(
_user_specified_nameconv2d_5_input
�
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_1456108

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
G
+__inference_flatten_3_layer_call_fn_1457440

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1456108a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1455930

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�
#__inference__traced_restore_1458071
file_prefix:
 assignvariableop_conv2d_5_kernel: .
 assignvariableop_1_conv2d_5_bias: <
.assignvariableop_2_batch_normalization_7_gamma: ;
-assignvariableop_3_batch_normalization_7_beta: B
4assignvariableop_4_batch_normalization_7_moving_mean: F
8assignvariableop_5_batch_normalization_7_moving_variance: <
"assignvariableop_6_conv2d_6_kernel: @.
 assignvariableop_7_conv2d_6_bias:@<
.assignvariableop_8_batch_normalization_8_gamma:@;
-assignvariableop_9_batch_normalization_8_beta:@C
5assignvariableop_10_batch_normalization_8_moving_mean:@G
9assignvariableop_11_batch_normalization_8_moving_variance:@5
"assignvariableop_12_dense_6_kernel:	�@.
 assignvariableop_13_dense_6_bias:@=
/assignvariableop_14_batch_normalization_9_gamma:@<
.assignvariableop_15_batch_normalization_9_beta:@C
5assignvariableop_16_batch_normalization_9_moving_mean:@G
9assignvariableop_17_batch_normalization_9_moving_variance:@4
"assignvariableop_18_dense_7_kernel:@.
 assignvariableop_19_dense_7_bias:&
assignvariableop_20_sgd_iter:	 '
assignvariableop_21_sgd_decay: /
%assignvariableop_22_sgd_learning_rate: *
 assignvariableop_23_sgd_momentum: %
assignvariableop_24_total_1: %
assignvariableop_25_count_1: #
assignvariableop_26_total: #
assignvariableop_27_count: J
0assignvariableop_28_sgd_conv2d_5_kernel_momentum: <
.assignvariableop_29_sgd_conv2d_5_bias_momentum: J
<assignvariableop_30_sgd_batch_normalization_7_gamma_momentum: I
;assignvariableop_31_sgd_batch_normalization_7_beta_momentum: J
0assignvariableop_32_sgd_conv2d_6_kernel_momentum: @<
.assignvariableop_33_sgd_conv2d_6_bias_momentum:@J
<assignvariableop_34_sgd_batch_normalization_8_gamma_momentum:@I
;assignvariableop_35_sgd_batch_normalization_8_beta_momentum:@B
/assignvariableop_36_sgd_dense_6_kernel_momentum:	�@;
-assignvariableop_37_sgd_dense_6_bias_momentum:@J
<assignvariableop_38_sgd_batch_normalization_9_gamma_momentum:@I
;assignvariableop_39_sgd_batch_normalization_9_beta_momentum:@A
/assignvariableop_40_sgd_dense_7_kernel_momentum:@;
-assignvariableop_41_sgd_dense_7_bias_momentum:
identity_43��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_conv2d_5_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_5_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_7_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_7_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_7_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_7_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_6_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_6_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_8_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_8_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_8_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_8_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_6_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_6_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_9_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_9_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_9_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_9_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_7_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_7_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_sgd_iterIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_sgd_decayIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp%assignvariableop_22_sgd_learning_rateIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp assignvariableop_23_sgd_momentumIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_1Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_1Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_totalIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_countIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp0assignvariableop_28_sgd_conv2d_5_kernel_momentumIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp.assignvariableop_29_sgd_conv2d_5_bias_momentumIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp<assignvariableop_30_sgd_batch_normalization_7_gamma_momentumIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp;assignvariableop_31_sgd_batch_normalization_7_beta_momentumIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp0assignvariableop_32_sgd_conv2d_6_kernel_momentumIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp.assignvariableop_33_sgd_conv2d_6_bias_momentumIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp<assignvariableop_34_sgd_batch_normalization_8_gamma_momentumIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp;assignvariableop_35_sgd_batch_normalization_8_beta_momentumIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp/assignvariableop_36_sgd_dense_6_kernel_momentumIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp-assignvariableop_37_sgd_dense_6_bias_momentumIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp<assignvariableop_38_sgd_batch_normalization_9_gamma_momentumIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp;assignvariableop_39_sgd_batch_normalization_9_beta_momentumIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp/assignvariableop_40_sgd_dense_7_kernel_momentumIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp-assignvariableop_41_sgd_dense_7_bias_momentumIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_43IdentityIdentity_42:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_43Identity_43:output:0*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
7__inference_batch_normalization_9_layer_call_fn_1457519

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1455971o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
+__inference_dropout_3_layer_call_fn_1457451

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1456122p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1456086

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
��
�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1457105

inputsA
'conv2d_5_conv2d_readvariableop_resource: 6
(conv2d_5_biasadd_readvariableop_resource: ;
-batch_normalization_7_readvariableop_resource: =
/batch_normalization_7_readvariableop_1_resource: L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_6_conv2d_readvariableop_resource: @6
(conv2d_6_biasadd_readvariableop_resource:@;
-batch_normalization_8_readvariableop_resource:@=
/batch_normalization_8_readvariableop_1_resource:@L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@9
&dense_6_matmul_readvariableop_resource:	�@5
'dense_6_biasadd_readvariableop_resource:@K
=batch_normalization_9_assignmovingavg_readvariableop_resource:@M
?batch_normalization_9_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_9_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_9_batchnorm_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@5
'dense_7_biasadd_readvariableop_resource:
identity��$batch_normalization_7/AssignNewValue�&batch_normalization_7/AssignNewValue_1�5batch_normalization_7/FusedBatchNormV3/ReadVariableOp�7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_7/ReadVariableOp�&batch_normalization_7/ReadVariableOp_1�$batch_normalization_8/AssignNewValue�&batch_normalization_8/AssignNewValue_1�5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_8/ReadVariableOp�&batch_normalization_8/ReadVariableOp_1�%batch_normalization_9/AssignMovingAvg�4batch_normalization_9/AssignMovingAvg/ReadVariableOp�'batch_normalization_9/AssignMovingAvg_1�6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_9/batchnorm/ReadVariableOp�2batch_normalization_9/batchnorm/mul/ReadVariableOp�conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�-dense_6/kernel/Regularizer/Abs/ReadVariableOp�0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������> *
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������> j
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:���������> �
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0�
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0�
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_5/Relu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������> : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
max_pooling2d_4/MaxPoolMaxPool*batch_normalization_7/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_6/Conv2DConv2D max_pooling2d_4/MaxPool:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_6/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
max_pooling2d_5/MaxPoolMaxPool*batch_normalization_8/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  �
flatten_3/ReshapeReshape max_pooling2d_5/MaxPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
dropout_3/dropout/MulMulflatten_3/Reshape:output:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:����������o
dropout_3/dropout/ShapeShapeflatten_3/Reshape:output:0*
T0*
_output_shapes
::���
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������^
dropout_3/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_3/dropout/SelectV2SelectV2"dropout_3/dropout/GreaterEqual:z:0dropout_3/dropout/Mul:z:0"dropout_3/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_6/MatMulMatMul#dropout_3/dropout/SelectV2:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������@~
4batch_normalization_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_9/moments/meanMeandense_6/Relu:activations:0=batch_normalization_9/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
*batch_normalization_9/moments/StopGradientStopGradient+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes

:@�
/batch_normalization_9/moments/SquaredDifferenceSquaredDifferencedense_6/Relu:activations:03batch_normalization_9/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
8batch_normalization_9/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_9/moments/varianceMean3batch_normalization_9/moments/SquaredDifference:z:0Abatch_normalization_9/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
%batch_normalization_9/moments/SqueezeSqueeze+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
'batch_normalization_9/moments/Squeeze_1Squeeze/batch_normalization_9/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 p
+batch_normalization_9/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_9_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
)batch_normalization_9/AssignMovingAvg/subSub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_9/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
)batch_normalization_9/AssignMovingAvg/mulMul-batch_normalization_9/AssignMovingAvg/sub:z:04batch_normalization_9/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
%batch_normalization_9/AssignMovingAvgAssignSubVariableOp=batch_normalization_9_assignmovingavg_readvariableop_resource-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_9/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_9_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
+batch_normalization_9/AssignMovingAvg_1/subSub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_9/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
+batch_normalization_9/AssignMovingAvg_1/mulMul/batch_normalization_9/AssignMovingAvg_1/sub:z:06batch_normalization_9/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
'batch_normalization_9/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_9_assignmovingavg_1_readvariableop_resource/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_9/batchnorm/addAddV20batch_normalization_9/moments/Squeeze_1:output:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:@�
2batch_normalization_9/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_9_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:0:batch_normalization_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
%batch_normalization_9/batchnorm/mul_1Muldense_6/Relu:activations:0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
%batch_normalization_9/batchnorm/mul_2Mul.batch_normalization_9/moments/Squeeze:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
.batch_normalization_9/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0�
#batch_normalization_9/batchnorm/subSub6batch_normalization_9/batchnorm/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_7/MatMulMatMul)batch_normalization_9/batchnorm/add_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_5/kernel/Regularizer/L2LossL2Loss9conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0+conv2d_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
-dense_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_6/kernel/Regularizer/AbsAbs5dense_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@s
"dense_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_6/kernel/Regularizer/SumSum"dense_6/kernel/Regularizer/Abs:y:0+dense_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_6/kernel/Regularizer/addAddV2)dense_6/kernel/Regularizer/Const:output:0"dense_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_6/kernel/Regularizer/mul_1Mul+dense_6/kernel/Regularizer/mul_1/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
 dense_6/kernel/Regularizer/add_1AddV2"dense_6/kernel/Regularizer/add:z:0$dense_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentitydense_7/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1&^batch_normalization_9/AssignMovingAvg5^batch_normalization_9/AssignMovingAvg/ReadVariableOp(^batch_normalization_9/AssignMovingAvg_17^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_9/batchnorm/ReadVariableOp3^batch_normalization_9/batchnorm/mul/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp.^dense_6/kernel/Regularizer/Abs/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������>: : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2l
4batch_normalization_9/AssignMovingAvg/ReadVariableOp4batch_normalization_9/AssignMovingAvg/ReadVariableOp2p
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization_9/AssignMovingAvg_1'batch_normalization_9/AssignMovingAvg_12N
%batch_normalization_9/AssignMovingAvg%batch_normalization_9/AssignMovingAvg2`
.batch_normalization_9/batchnorm/ReadVariableOp.batch_normalization_9/batchnorm/ReadVariableOp2h
2batch_normalization_9/batchnorm/mul/ReadVariableOp2batch_normalization_9/batchnorm/mul/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2^
-dense_6/kernel/Regularizer/Abs/ReadVariableOp-dense_6/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1457225

inputsA
'conv2d_5_conv2d_readvariableop_resource: 6
(conv2d_5_biasadd_readvariableop_resource: ;
-batch_normalization_7_readvariableop_resource: =
/batch_normalization_7_readvariableop_1_resource: L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_6_conv2d_readvariableop_resource: @6
(conv2d_6_biasadd_readvariableop_resource:@;
-batch_normalization_8_readvariableop_resource:@=
/batch_normalization_8_readvariableop_1_resource:@L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@9
&dense_6_matmul_readvariableop_resource:	�@5
'dense_6_biasadd_readvariableop_resource:@E
7batch_normalization_9_batchnorm_readvariableop_resource:@I
;batch_normalization_9_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_9_batchnorm_readvariableop_1_resource:@G
9batch_normalization_9_batchnorm_readvariableop_2_resource:@8
&dense_7_matmul_readvariableop_resource:@5
'dense_7_biasadd_readvariableop_resource:
identity��5batch_normalization_7/FusedBatchNormV3/ReadVariableOp�7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_7/ReadVariableOp�&batch_normalization_7/ReadVariableOp_1�5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_8/ReadVariableOp�&batch_normalization_8/ReadVariableOp_1�.batch_normalization_9/batchnorm/ReadVariableOp�0batch_normalization_9/batchnorm/ReadVariableOp_1�0batch_normalization_9/batchnorm/ReadVariableOp_2�2batch_normalization_9/batchnorm/mul/ReadVariableOp�conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�-dense_6/kernel/Regularizer/Abs/ReadVariableOp�0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������> *
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������> j
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:���������> �
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0�
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0�
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_5/Relu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������> : : : : :*
epsilon%o�:*
is_training( �
max_pooling2d_4/MaxPoolMaxPool*batch_normalization_7/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_6/Conv2DConv2D max_pooling2d_4/MaxPool:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_6/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
max_pooling2d_5/MaxPoolMaxPool*batch_normalization_8/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  �
flatten_3/ReshapeReshape max_pooling2d_5/MaxPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������m
dropout_3/IdentityIdentityflatten_3/Reshape:output:0*
T0*(
_output_shapes
:�����������
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_6/MatMulMatMuldropout_3/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
.batch_normalization_9/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_9/batchnorm/addAddV26batch_normalization_9/batchnorm/ReadVariableOp:value:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:@�
2batch_normalization_9/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_9_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:0:batch_normalization_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
%batch_normalization_9/batchnorm/mul_1Muldense_6/Relu:activations:0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
0batch_normalization_9/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_9_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
%batch_normalization_9/batchnorm/mul_2Mul8batch_normalization_9/batchnorm/ReadVariableOp_1:value:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
0batch_normalization_9/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_9_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
#batch_normalization_9/batchnorm/subSub8batch_normalization_9/batchnorm/ReadVariableOp_2:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_7/MatMulMatMul)batch_normalization_9/batchnorm/add_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_5/kernel/Regularizer/L2LossL2Loss9conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0+conv2d_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
-dense_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_6/kernel/Regularizer/AbsAbs5dense_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@s
"dense_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_6/kernel/Regularizer/SumSum"dense_6/kernel/Regularizer/Abs:y:0+dense_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_6/kernel/Regularizer/addAddV2)dense_6/kernel/Regularizer/Const:output:0"dense_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_6/kernel/Regularizer/mul_1Mul+dense_6/kernel/Regularizer/mul_1/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
 dense_6/kernel/Regularizer/add_1AddV2"dense_6/kernel/Regularizer/add:z:0$dense_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentitydense_7/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp6^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1/^batch_normalization_9/batchnorm/ReadVariableOp1^batch_normalization_9/batchnorm/ReadVariableOp_11^batch_normalization_9/batchnorm/ReadVariableOp_23^batch_normalization_9/batchnorm/mul/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp.^dense_6/kernel/Regularizer/Abs/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������>: : : : : : : : : : : : : : : : : : : : 2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2d
0batch_normalization_9/batchnorm/ReadVariableOp_10batch_normalization_9/batchnorm/ReadVariableOp_12d
0batch_normalization_9/batchnorm/ReadVariableOp_20batch_normalization_9/batchnorm/ReadVariableOp_22`
.batch_normalization_9/batchnorm/ReadVariableOp.batch_normalization_9/batchnorm/ReadVariableOp2h
2batch_normalization_9/batchnorm/mul/ReadVariableOp2batch_normalization_9/batchnorm/mul/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2^
-dense_6/kernel/Regularizer/Abs/ReadVariableOp-dense_6/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1457302

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1457320

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�l
�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1456416

inputs*
conv2d_5_1456325: 
conv2d_5_1456327: +
batch_normalization_7_1456330: +
batch_normalization_7_1456332: +
batch_normalization_7_1456334: +
batch_normalization_7_1456336: *
conv2d_6_1456340: @
conv2d_6_1456342:@+
batch_normalization_8_1456345:@+
batch_normalization_8_1456347:@+
batch_normalization_8_1456349:@+
batch_normalization_8_1456351:@"
dense_6_1456357:	�@
dense_6_1456359:@+
batch_normalization_9_1456362:@+
batch_normalization_9_1456364:@+
batch_normalization_9_1456366:@+
batch_normalization_9_1456368:@!
dense_7_1456371:@
dense_7_1456373:
identity��-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp� conv2d_6/StatefulPartitionedCall�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense_6/StatefulPartitionedCall�-dense_6/kernel/Regularizer/Abs/ReadVariableOp�0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp�dense_7/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_1456325conv2d_5_1456327*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������> *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1456046�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_7_1456330batch_normalization_7_1456332batch_normalization_7_1456334batch_normalization_7_1456336*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������> *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1455803�
max_pooling2d_4/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1455854�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_6_1456340conv2d_6_1456342*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1456086�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_8_1456345batch_normalization_8_1456347batch_normalization_8_1456349batch_normalization_8_1456351*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1455879�
max_pooling2d_5/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1455930�
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_1456108�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1456122�
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_6_1456357dense_6_1456359*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1456148�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_9_1456362batch_normalization_9_1456364batch_normalization_9_1456366batch_normalization_9_1456368*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1455971�
dense_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0dense_7_1456371dense_7_1456373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1456174f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_5_1456325*&
_output_shapes
: *
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_5_1456325*&
_output_shapes
: *
dtype0�
"conv2d_5/kernel/Regularizer/L2LossL2Loss9conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0+conv2d_5/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_6_1456340*&
_output_shapes
: @*
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_6_1456340*&
_output_shapes
: @*
dtype0�
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
-dense_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_6_1456357*
_output_shapes
:	�@*
dtype0�
dense_6/kernel/Regularizer/AbsAbs5dense_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@s
"dense_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_6/kernel/Regularizer/SumSum"dense_6/kernel/Regularizer/Abs:y:0+dense_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_6/kernel/Regularizer/addAddV2)dense_6/kernel/Regularizer/Const:output:0"dense_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_1456357*
_output_shapes
:	�@*
dtype0�
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_6/kernel/Regularizer/mul_1Mul+dense_6/kernel/Regularizer/mul_1/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
 dense_6/kernel/Regularizer/add_1AddV2"dense_6/kernel/Regularizer/add:z:0$dense_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_6/StatefulPartitionedCall.^dense_6/kernel/Regularizer/Abs/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_7/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������>: : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_5/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2^
-dense_6/kernel/Regularizer/Abs/ReadVariableOp-dense_6/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:W S
/
_output_shapes
:���������>
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
Q
conv2d_5_input?
 serving_default_conv2d_5_input:0���������>;
dense_70
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$axis
	%gamma
&beta
'moving_mean
(moving_variance"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias
 7_jit_compiled_convolution_op"
_tf_keras_layer
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
U_random_generator"
_tf_keras_layer
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance"
_tf_keras_layer
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

okernel
pbias"
_tf_keras_layer
�
0
1
%2
&3
'4
(5
56
67
?8
@9
A10
B11
\12
]13
e14
f15
g16
h17
o18
p19"
trackable_list_wrapper
�
0
1
%2
&3
54
65
?6
@7
\8
]9
e10
f11
o12
p13"
trackable_list_wrapper
5
q0
r1
s2"
trackable_list_wrapper
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
ytrace_0
ztrace_1
{trace_2
|trace_32�
.__inference_sequential_3_layer_call_fn_1456459
.__inference_sequential_3_layer_call_fn_1456598
.__inference_sequential_3_layer_call_fn_1456919
.__inference_sequential_3_layer_call_fn_1456964�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0zztrace_1z{trace_2z|trace_3
�
}trace_0
~trace_1
trace_2
�trace_32�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1456220
I__inference_sequential_3_layer_call_and_return_conditional_losses_1456319
I__inference_sequential_3_layer_call_and_return_conditional_losses_1457105
I__inference_sequential_3_layer_call_and_return_conditional_losses_1457225�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z}trace_0z~trace_1ztrace_2z�trace_3
�B�
"__inference__wrapped_model_1455784conv2d_5_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	�iter

�decay
�learning_rate
�momentummomentum�momentum�%momentum�&momentum�5momentum�6momentum�?momentum�@momentum�\momentum�]momentum�emomentum�fmomentum�omomentum�pmomentum�"
	optimizer
-
�serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
q0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_5_layer_call_fn_1457234�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1457258�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):' 2conv2d_5/kernel
: 2conv2d_5/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
%0
&1
'2
(3"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_7_layer_call_fn_1457271
7__inference_batch_normalization_7_layer_call_fn_1457284�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1457302
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1457320�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
):' 2batch_normalization_7/gamma
(:& 2batch_normalization_7/beta
1:/  (2!batch_normalization_7/moving_mean
5:3  (2%batch_normalization_7/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_4_layer_call_fn_1457325�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1457330�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
'
r0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_6_layer_call_fn_1457339�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1457363�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):' @2conv2d_6/kernel
:@2conv2d_6/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
?0
@1
A2
B3"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_8_layer_call_fn_1457376
7__inference_batch_normalization_8_layer_call_fn_1457389�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1457407
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1457425�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
):'@2batch_normalization_8/gamma
(:&@2batch_normalization_8/beta
1:/@ (2!batch_normalization_8/moving_mean
5:3@ (2%batch_normalization_8/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_5_layer_call_fn_1457430�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1457435�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_flatten_3_layer_call_fn_1457440�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_flatten_3_layer_call_and_return_conditional_losses_1457446�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_3_layer_call_fn_1457451
+__inference_dropout_3_layer_call_fn_1457456�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_3_layer_call_and_return_conditional_losses_1457468
F__inference_dropout_3_layer_call_and_return_conditional_losses_1457473�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
'
s0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_6_layer_call_fn_1457482�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_6_layer_call_and_return_conditional_losses_1457506�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:	�@2dense_6/kernel
:@2dense_6/bias
<
e0
f1
g2
h3"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_9_layer_call_fn_1457519
7__inference_batch_normalization_9_layer_call_fn_1457532�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1457566
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1457586�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
):'@2batch_normalization_9/gamma
(:&@2batch_normalization_9/beta
1:/@ (2!batch_normalization_9/moving_mean
5:3@ (2%batch_normalization_9/moving_variance
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_7_layer_call_fn_1457595�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_7_layer_call_and_return_conditional_losses_1457606�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :@2dense_7/kernel
:2dense_7/bias
�
�trace_02�
__inference_loss_fn_0_1457624�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_1457642�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_1457660�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
J
'0
(1
A2
B3
g4
h5"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_sequential_3_layer_call_fn_1456459conv2d_5_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_3_layer_call_fn_1456598conv2d_5_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_3_layer_call_fn_1456919inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_3_layer_call_fn_1456964inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1456220conv2d_5_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1456319conv2d_5_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1457105inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1457225inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
�B�
%__inference_signature_wrapper_1456835conv2d_5_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
q0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_5_layer_call_fn_1457234inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1457258inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_batch_normalization_7_layer_call_fn_1457271inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_batch_normalization_7_layer_call_fn_1457284inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1457302inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1457320inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_max_pooling2d_4_layer_call_fn_1457325inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1457330inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
r0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_6_layer_call_fn_1457339inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1457363inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_batch_normalization_8_layer_call_fn_1457376inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_batch_normalization_8_layer_call_fn_1457389inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1457407inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1457425inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_max_pooling2d_5_layer_call_fn_1457430inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1457435inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_flatten_3_layer_call_fn_1457440inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_flatten_3_layer_call_and_return_conditional_losses_1457446inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dropout_3_layer_call_fn_1457451inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_3_layer_call_fn_1457456inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_3_layer_call_and_return_conditional_losses_1457468inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_3_layer_call_and_return_conditional_losses_1457473inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
s0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_6_layer_call_fn_1457482inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_6_layer_call_and_return_conditional_losses_1457506inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_batch_normalization_9_layer_call_fn_1457519inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_batch_normalization_9_layer_call_fn_1457532inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1457566inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1457586inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_7_layer_call_fn_1457595inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_7_layer_call_and_return_conditional_losses_1457606inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_1457624"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_1457642"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_1457660"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
4:2 2SGD/conv2d_5/kernel/momentum
&:$ 2SGD/conv2d_5/bias/momentum
4:2 2(SGD/batch_normalization_7/gamma/momentum
3:1 2'SGD/batch_normalization_7/beta/momentum
4:2 @2SGD/conv2d_6/kernel/momentum
&:$@2SGD/conv2d_6/bias/momentum
4:2@2(SGD/batch_normalization_8/gamma/momentum
3:1@2'SGD/batch_normalization_8/beta/momentum
,:*	�@2SGD/dense_6/kernel/momentum
%:#@2SGD/dense_6/bias/momentum
4:2@2(SGD/batch_normalization_9/gamma/momentum
3:1@2'SGD/batch_normalization_9/beta/momentum
+:)@2SGD/dense_7/kernel/momentum
%:#2SGD/dense_7/bias/momentum�
"__inference__wrapped_model_1455784�%&'(56?@AB\]hegfop?�<
5�2
0�-
conv2d_5_input���������>
� "1�.
,
dense_7!�
dense_7����������
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1457302�%&'(Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1457320�%&'(Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
7__inference_batch_normalization_7_layer_call_fn_1457271�%&'(Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� ";�8
unknown+��������������������������� �
7__inference_batch_normalization_7_layer_call_fn_1457284�%&'(Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� ";�8
unknown+��������������������������� �
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1457407�?@ABQ�N
G�D
:�7
inputs+���������������������������@
p

 
� "F�C
<�9
tensor_0+���������������������������@
� �
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1457425�?@ABQ�N
G�D
:�7
inputs+���������������������������@
p 

 
� "F�C
<�9
tensor_0+���������������������������@
� �
7__inference_batch_normalization_8_layer_call_fn_1457376�?@ABQ�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
7__inference_batch_normalization_8_layer_call_fn_1457389�?@ABQ�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1457566mghef7�4
-�*
 �
inputs���������@
p

 
� ",�)
"�
tensor_0���������@
� �
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1457586mhegf7�4
-�*
 �
inputs���������@
p 

 
� ",�)
"�
tensor_0���������@
� �
7__inference_batch_normalization_9_layer_call_fn_1457519bghef7�4
-�*
 �
inputs���������@
p

 
� "!�
unknown���������@�
7__inference_batch_normalization_9_layer_call_fn_1457532bhegf7�4
-�*
 �
inputs���������@
p 

 
� "!�
unknown���������@�
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1457258s7�4
-�*
(�%
inputs���������>
� "4�1
*�'
tensor_0���������> 
� �
*__inference_conv2d_5_layer_call_fn_1457234h7�4
-�*
(�%
inputs���������>
� ")�&
unknown���������> �
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1457363s567�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0���������@
� �
*__inference_conv2d_6_layer_call_fn_1457339h567�4
-�*
(�%
inputs��������� 
� ")�&
unknown���������@�
D__inference_dense_6_layer_call_and_return_conditional_losses_1457506d\]0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
)__inference_dense_6_layer_call_fn_1457482Y\]0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
D__inference_dense_7_layer_call_and_return_conditional_losses_1457606cop/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������
� �
)__inference_dense_7_layer_call_fn_1457595Xop/�,
%�"
 �
inputs���������@
� "!�
unknown����������
F__inference_dropout_3_layer_call_and_return_conditional_losses_1457468e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
F__inference_dropout_3_layer_call_and_return_conditional_losses_1457473e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
+__inference_dropout_3_layer_call_fn_1457451Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
+__inference_dropout_3_layer_call_fn_1457456Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
F__inference_flatten_3_layer_call_and_return_conditional_losses_1457446h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
tensor_0����������
� �
+__inference_flatten_3_layer_call_fn_1457440]7�4
-�*
(�%
inputs���������@
� ""�
unknown����������E
__inference_loss_fn_0_1457624$�

� 
� "�
unknown E
__inference_loss_fn_1_1457642$5�

� 
� "�
unknown E
__inference_loss_fn_2_1457660$\�

� 
� "�
unknown �
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1457330�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_max_pooling2d_4_layer_call_fn_1457325�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1457435�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_max_pooling2d_5_layer_call_fn_1457430�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
I__inference_sequential_3_layer_call_and_return_conditional_losses_1456220�%&'(56?@AB\]ghefopG�D
=�:
0�-
conv2d_5_input���������>
p

 
� ",�)
"�
tensor_0���������
� �
I__inference_sequential_3_layer_call_and_return_conditional_losses_1456319�%&'(56?@AB\]hegfopG�D
=�:
0�-
conv2d_5_input���������>
p 

 
� ",�)
"�
tensor_0���������
� �
I__inference_sequential_3_layer_call_and_return_conditional_losses_1457105�%&'(56?@AB\]ghefop?�<
5�2
(�%
inputs���������>
p

 
� ",�)
"�
tensor_0���������
� �
I__inference_sequential_3_layer_call_and_return_conditional_losses_1457225�%&'(56?@AB\]hegfop?�<
5�2
(�%
inputs���������>
p 

 
� ",�)
"�
tensor_0���������
� �
.__inference_sequential_3_layer_call_fn_1456459�%&'(56?@AB\]ghefopG�D
=�:
0�-
conv2d_5_input���������>
p

 
� "!�
unknown����������
.__inference_sequential_3_layer_call_fn_1456598�%&'(56?@AB\]hegfopG�D
=�:
0�-
conv2d_5_input���������>
p 

 
� "!�
unknown����������
.__inference_sequential_3_layer_call_fn_1456919z%&'(56?@AB\]ghefop?�<
5�2
(�%
inputs���������>
p

 
� "!�
unknown����������
.__inference_sequential_3_layer_call_fn_1456964z%&'(56?@AB\]hegfop?�<
5�2
(�%
inputs���������>
p 

 
� "!�
unknown����������
%__inference_signature_wrapper_1456835�%&'(56?@AB\]hegfopQ�N
� 
G�D
B
conv2d_5_input0�-
conv2d_5_input���������>"1�.
,
dense_7!�
dense_7���������