�(
��
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
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
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
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
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
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
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
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
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��"
�
ConstConst*"
_output_shapes
:>*
dtype0*�
value�B�>"�      �?      �?      �?      �?      �?      �?�jW?@Q
?R�Z>�z?>=k�?W�#<��?c1;��?YZ�9��?��h?3վ���>��h?�ؽ=��~?>ԣ<��?N1�;d�?XZs:��?Á>&p}��-?V]L?!>��}? ��<��?���;��?���:��?ϽA�0U'��MB?��&?s=>��{?��#=��?�0<��?QZ�:��?|u�,<�>�qa?c��>��k>�"y?��L=�?�|0<3�?o;��?�����u?c)v?���>���>�"v?ќu=�?��S<��?��6;��?F0(?��@?A?)O�=jq�>�r?2>�=�_?w<��?��T;��?�F}?����}?U���ǹ>��n?Q��=f.?�/�<D�?5Zs;��?2�>�?i���n?�H�����>\�i??�=��~?՞<��?��;n�?�D�d�V�a�U?��5�>0�d?vu�=��~?Fz�<��?T�;K�?\��r�;��2?�z7�4�>�Y_?r��=�s~?L�<��?�M�;%�?�\	��X?�,?#gY��T?8NY?�+�=�(~?��<�?���;��?" �>oNh?%�>CFq�tD?��R?о> �}?�h�<L�?��;��?r�}?�>� >��}�;�?0�K?d�>�~}?��<1�?���;��?Dy&?�zB�@1��i�~��.$?)kD?O> }?nX=��?;$�;i�?h��,)u�����i%t�n-?і<?P$#>��|?B*=�?�Y�;1�?%v�6⌾�����
^�̮5?zZ4?#>->IO|?��=�?�G<��?�@@��
)? �+��=���=?��+?�S7>.�{?g�=��?j�<��?iy>�}?ˁP�d��x�E?2�"?:dA>�d{?��'=�?+}<t�?ȶi?"��>upk�n
ɾ��L?�c?�oK>��z?�o0=,�?�<-�?/V?�7�S|{��t?�y�S?(�?�vU><`z?�@9=�?��<��?�o��v��N*�<%KZ?�?�w_>j�y?WB=f�?_M'<��?�X��g��}x��3v>pC`?���>�ri>3By?��J=��?�.<D�?��g��-�>f�e���>�e?���>Qhs>��x?��S=j�?̂6<��?8��x�}?�H���? �j?F�>wW}>�
x?�\=��?><��?7C?��%?�L!���F?}:o?�E�>��>Uew?�Qe=6�?.�E<:�?�t?8����%�׭d?3s?��>��>��v?� n='�?�RM<��?*��>Wmv��~�,�w?��v?u"�>}~�>�v?�v=ˈ?��T<w�?��)���?����?��y?k2d>Th�>xOu?��=!�?-�\<�?��|��>(r7>��{?��{?�5>mN�>�t?�F�=)w?�"d<��?�ξ�,j?�I�>Q;l?��}?��>�0�>"�s?٭�=�m?r�k<7�?M*?��U?r�?��Q?�?�ݮ=��>s?��=Qd?Xs<��?9�?��Y��K<?p-?��?��=��>�/r?�{�=pZ?��z<P�?�q?�;Y�6]?�,?<�?Y��Ҿ�>_Xq?+�=BP?�F�<��?�:۾_Xg���s?S��>R�?\�M��>�zp?�H�=�E?��<Z�?8�}���q�~?�j�=q�~?�̽J]�>�o?֮�=�:?1�<��?�$�-�C?�<~?��A}?����%�>�n?��=�/?w��<V�?���>�t?��q?�L��?{?1jD�J�>��m?�z�=$?�{�<��?K�v?l��>�xZ?�p���x?w�r���>��l?i�=�?�H�<C�?��>?t�*���8?P1��u?j]���a�>��k?�E�=�?;�<��?�m"�D�|���?G~T���q?~ ����>n�j?��=~ ?y�<"�?(�j���̾��>n�8�m?yG�����>2�i?%�=��~?���<��?1�T��?��#>��|�}7i?&Ӿp�>��h?�t�=��~?�}�<��?��<��?�`�۝�'d?B�辽�>Νg?���=��~?$K�<U�?��Y?�{?������v�m^?/z��ͳ�>��f?�=�=?�~?X�<��?��f?mFݾ���gb�LNX?+��+M�>�ce?��=n�~?��<�?��=�	~�� %���C�D�Q?������>�=d?��=O�~?���<h�?V�D���#��!K������J?[b�Rn�>c?�i�=�~?��<��?")t�D�>�g���پ#5C?��%����>��a?���=(�~?M�<�?�U��w?h�y�F�b�P;?�.�Dw�>@�`?0�=!�~?8�<Z�?$�+?��=?���E_��3?� 7�^��>/l_?��=�t~?]��<��?��|?�&�`�z���R>�T*?�?�g�>f)^?���=*e~?��<��?���>�k��qi�\$�>�G!?�F��j?��\?�W�=:U~?���<,�?���MT���M�:�?��?�N�k�?̒[?��=�D~?�N�<j�?���,C�<o8(���@?�'?3�T���??Z?��=s4~?��<��?^���lZ?�'���x`?�?E[���?��X?"}�=�#~?���<��?�P�>�]f?Q}��6�u?���>�)a�S%
?ӆW?R��=v~?���<�?�,~?��=Ց���\?`u�>Œf�LK?k"V?5?�=~?��<A�?C#?�eE�g�>5R}?r��>f|k��m?��T?��>E�}?P�<m�?#��V�s�mz�>s�o?���>,�o�i�?2IS?	 >8�}?(�<��?Sw��%��4^?�V?�L�>��s�q�?t�Q?0>��}?/��<��?
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
�
Adam/v/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_22/bias
y
(Adam/v/dense_22/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_22/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_22/bias
y
(Adam/m/dense_22/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_22/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_22/kernel
�
*Adam/v/dense_22/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_22/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_22/kernel
�
*Adam/m/dense_22/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_22/kernel*
_output_shapes

:*
dtype0
�
"Adam/v/layer_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_15/beta
�
6Adam/v/layer_normalization_15/beta/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_15/beta*
_output_shapes
:*
dtype0
�
"Adam/m/layer_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_15/beta
�
6Adam/m/layer_normalization_15/beta/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_15/beta*
_output_shapes
:*
dtype0
�
#Adam/v/layer_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/layer_normalization_15/gamma
�
7Adam/v/layer_normalization_15/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/layer_normalization_15/gamma*
_output_shapes
:*
dtype0
�
#Adam/m/layer_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/layer_normalization_15/gamma
�
7Adam/m/layer_normalization_15/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/layer_normalization_15/gamma*
_output_shapes
:*
dtype0
�
Adam/v/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_21/bias
y
(Adam/v/dense_21/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_21/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_21/bias
y
(Adam/m/dense_21/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_21/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_21/kernel
�
*Adam/v/dense_21/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_21/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_21/kernel
�
*Adam/m/dense_21/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_21/kernel*
_output_shapes

:*
dtype0
�
!Adam/v/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/batch_normalization_1/beta
�
5Adam/v/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_1/beta*
_output_shapes
:*
dtype0
�
!Adam/m/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/batch_normalization_1/beta
�
5Adam/m/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_1/beta*
_output_shapes
:*
dtype0
�
"Adam/v/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_1/gamma
�
6Adam/v/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_1/gamma*
_output_shapes
:*
dtype0
�
"Adam/m/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_1/gamma
�
6Adam/m/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_1/gamma*
_output_shapes
:*
dtype0
�
Adam/v/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_20/kernel
�
*Adam/v/dense_20/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_20/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_20/kernel
�
*Adam/m/dense_20/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_20/kernel*
_output_shapes

:*
dtype0
�
"Adam/v/layer_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_14/beta
�
6Adam/v/layer_normalization_14/beta/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_14/beta*
_output_shapes
:*
dtype0
�
"Adam/m/layer_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_14/beta
�
6Adam/m/layer_normalization_14/beta/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_14/beta*
_output_shapes
:*
dtype0
�
#Adam/v/layer_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/layer_normalization_14/gamma
�
7Adam/v/layer_normalization_14/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/layer_normalization_14/gamma*
_output_shapes
:*
dtype0
�
#Adam/m/layer_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/layer_normalization_14/gamma
�
7Adam/m/layer_normalization_14/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/layer_normalization_14/gamma*
_output_shapes
:*
dtype0
�
3Adam/v/multi_head_attention_7/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/v/multi_head_attention_7/attention_output/bias
�
GAdam/v/multi_head_attention_7/attention_output/bias/Read/ReadVariableOpReadVariableOp3Adam/v/multi_head_attention_7/attention_output/bias*
_output_shapes
:*
dtype0
�
3Adam/m/multi_head_attention_7/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/m/multi_head_attention_7/attention_output/bias
�
GAdam/m/multi_head_attention_7/attention_output/bias/Read/ReadVariableOpReadVariableOp3Adam/m/multi_head_attention_7/attention_output/bias*
_output_shapes
:*
dtype0
�
5Adam/v/multi_head_attention_7/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/v/multi_head_attention_7/attention_output/kernel
�
IAdam/v/multi_head_attention_7/attention_output/kernel/Read/ReadVariableOpReadVariableOp5Adam/v/multi_head_attention_7/attention_output/kernel*"
_output_shapes
:*
dtype0
�
5Adam/m/multi_head_attention_7/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/m/multi_head_attention_7/attention_output/kernel
�
IAdam/m/multi_head_attention_7/attention_output/kernel/Read/ReadVariableOpReadVariableOp5Adam/m/multi_head_attention_7/attention_output/kernel*"
_output_shapes
:*
dtype0
�
(Adam/v/multi_head_attention_7/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(Adam/v/multi_head_attention_7/value/bias
�
<Adam/v/multi_head_attention_7/value/bias/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_7/value/bias*
_output_shapes

:*
dtype0
�
(Adam/m/multi_head_attention_7/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(Adam/m/multi_head_attention_7/value/bias
�
<Adam/m/multi_head_attention_7/value/bias/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_7/value/bias*
_output_shapes

:*
dtype0
�
*Adam/v/multi_head_attention_7/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/v/multi_head_attention_7/value/kernel
�
>Adam/v/multi_head_attention_7/value/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/multi_head_attention_7/value/kernel*"
_output_shapes
:*
dtype0
�
*Adam/m/multi_head_attention_7/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/m/multi_head_attention_7/value/kernel
�
>Adam/m/multi_head_attention_7/value/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/multi_head_attention_7/value/kernel*"
_output_shapes
:*
dtype0
�
&Adam/v/multi_head_attention_7/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/v/multi_head_attention_7/key/bias
�
:Adam/v/multi_head_attention_7/key/bias/Read/ReadVariableOpReadVariableOp&Adam/v/multi_head_attention_7/key/bias*
_output_shapes

:*
dtype0
�
&Adam/m/multi_head_attention_7/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/m/multi_head_attention_7/key/bias
�
:Adam/m/multi_head_attention_7/key/bias/Read/ReadVariableOpReadVariableOp&Adam/m/multi_head_attention_7/key/bias*
_output_shapes

:*
dtype0
�
(Adam/v/multi_head_attention_7/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/v/multi_head_attention_7/key/kernel
�
<Adam/v/multi_head_attention_7/key/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_7/key/kernel*"
_output_shapes
:*
dtype0
�
(Adam/m/multi_head_attention_7/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/m/multi_head_attention_7/key/kernel
�
<Adam/m/multi_head_attention_7/key/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_7/key/kernel*"
_output_shapes
:*
dtype0
�
(Adam/v/multi_head_attention_7/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(Adam/v/multi_head_attention_7/query/bias
�
<Adam/v/multi_head_attention_7/query/bias/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_7/query/bias*
_output_shapes

:*
dtype0
�
(Adam/m/multi_head_attention_7/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(Adam/m/multi_head_attention_7/query/bias
�
<Adam/m/multi_head_attention_7/query/bias/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_7/query/bias*
_output_shapes

:*
dtype0
�
*Adam/v/multi_head_attention_7/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/v/multi_head_attention_7/query/kernel
�
>Adam/v/multi_head_attention_7/query/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/multi_head_attention_7/query/kernel*"
_output_shapes
:*
dtype0
�
*Adam/m/multi_head_attention_7/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/m/multi_head_attention_7/query/kernel
�
>Adam/m/multi_head_attention_7/query/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/multi_head_attention_7/query/kernel*"
_output_shapes
:*
dtype0
�
"Adam/v/layer_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_13/beta
�
6Adam/v/layer_normalization_13/beta/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_13/beta*
_output_shapes
:*
dtype0
�
"Adam/m/layer_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_13/beta
�
6Adam/m/layer_normalization_13/beta/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_13/beta*
_output_shapes
:*
dtype0
�
#Adam/v/layer_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/layer_normalization_13/gamma
�
7Adam/v/layer_normalization_13/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/layer_normalization_13/gamma*
_output_shapes
:*
dtype0
�
#Adam/m/layer_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/layer_normalization_13/gamma
�
7Adam/m/layer_normalization_13/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/layer_normalization_13/gamma*
_output_shapes
:*
dtype0
�
Adam/v/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_19/bias
y
(Adam/v/dense_19/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_19/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_19/bias
y
(Adam/m/dense_19/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_19/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_19/kernel
�
*Adam/v/dense_19/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_19/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_19/kernel
�
*Adam/m/dense_19/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_19/kernel*
_output_shapes

:*
dtype0
�
Adam/v/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/v/batch_normalization/beta
�
3Adam/v/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/v/batch_normalization/beta*
_output_shapes
:*
dtype0
�
Adam/m/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/m/batch_normalization/beta
�
3Adam/m/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/m/batch_normalization/beta*
_output_shapes
:*
dtype0
�
 Adam/v/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/v/batch_normalization/gamma
�
4Adam/v/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/v/batch_normalization/gamma*
_output_shapes
:*
dtype0
�
 Adam/m/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/m/batch_normalization/gamma
�
4Adam/m/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/m/batch_normalization/gamma*
_output_shapes
:*
dtype0
�
Adam/v/dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_18/kernel
�
*Adam/v/dense_18/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_18/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_18/kernel
�
*Adam/m/dense_18/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_18/kernel*
_output_shapes

:*
dtype0
�
"Adam/v/layer_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/layer_normalization_12/beta
�
6Adam/v/layer_normalization_12/beta/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_12/beta*
_output_shapes
:*
dtype0
�
"Adam/m/layer_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/layer_normalization_12/beta
�
6Adam/m/layer_normalization_12/beta/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_12/beta*
_output_shapes
:*
dtype0
�
#Adam/v/layer_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/layer_normalization_12/gamma
�
7Adam/v/layer_normalization_12/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/layer_normalization_12/gamma*
_output_shapes
:*
dtype0
�
#Adam/m/layer_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/layer_normalization_12/gamma
�
7Adam/m/layer_normalization_12/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/layer_normalization_12/gamma*
_output_shapes
:*
dtype0
�
3Adam/v/multi_head_attention_6/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/v/multi_head_attention_6/attention_output/bias
�
GAdam/v/multi_head_attention_6/attention_output/bias/Read/ReadVariableOpReadVariableOp3Adam/v/multi_head_attention_6/attention_output/bias*
_output_shapes
:*
dtype0
�
3Adam/m/multi_head_attention_6/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/m/multi_head_attention_6/attention_output/bias
�
GAdam/m/multi_head_attention_6/attention_output/bias/Read/ReadVariableOpReadVariableOp3Adam/m/multi_head_attention_6/attention_output/bias*
_output_shapes
:*
dtype0
�
5Adam/v/multi_head_attention_6/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/v/multi_head_attention_6/attention_output/kernel
�
IAdam/v/multi_head_attention_6/attention_output/kernel/Read/ReadVariableOpReadVariableOp5Adam/v/multi_head_attention_6/attention_output/kernel*"
_output_shapes
:*
dtype0
�
5Adam/m/multi_head_attention_6/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/m/multi_head_attention_6/attention_output/kernel
�
IAdam/m/multi_head_attention_6/attention_output/kernel/Read/ReadVariableOpReadVariableOp5Adam/m/multi_head_attention_6/attention_output/kernel*"
_output_shapes
:*
dtype0
�
(Adam/v/multi_head_attention_6/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(Adam/v/multi_head_attention_6/value/bias
�
<Adam/v/multi_head_attention_6/value/bias/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_6/value/bias*
_output_shapes

:*
dtype0
�
(Adam/m/multi_head_attention_6/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(Adam/m/multi_head_attention_6/value/bias
�
<Adam/m/multi_head_attention_6/value/bias/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_6/value/bias*
_output_shapes

:*
dtype0
�
*Adam/v/multi_head_attention_6/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/v/multi_head_attention_6/value/kernel
�
>Adam/v/multi_head_attention_6/value/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/multi_head_attention_6/value/kernel*"
_output_shapes
:*
dtype0
�
*Adam/m/multi_head_attention_6/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/m/multi_head_attention_6/value/kernel
�
>Adam/m/multi_head_attention_6/value/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/multi_head_attention_6/value/kernel*"
_output_shapes
:*
dtype0
�
&Adam/v/multi_head_attention_6/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/v/multi_head_attention_6/key/bias
�
:Adam/v/multi_head_attention_6/key/bias/Read/ReadVariableOpReadVariableOp&Adam/v/multi_head_attention_6/key/bias*
_output_shapes

:*
dtype0
�
&Adam/m/multi_head_attention_6/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/m/multi_head_attention_6/key/bias
�
:Adam/m/multi_head_attention_6/key/bias/Read/ReadVariableOpReadVariableOp&Adam/m/multi_head_attention_6/key/bias*
_output_shapes

:*
dtype0
�
(Adam/v/multi_head_attention_6/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/v/multi_head_attention_6/key/kernel
�
<Adam/v/multi_head_attention_6/key/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_6/key/kernel*"
_output_shapes
:*
dtype0
�
(Adam/m/multi_head_attention_6/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/m/multi_head_attention_6/key/kernel
�
<Adam/m/multi_head_attention_6/key/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_6/key/kernel*"
_output_shapes
:*
dtype0
�
(Adam/v/multi_head_attention_6/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(Adam/v/multi_head_attention_6/query/bias
�
<Adam/v/multi_head_attention_6/query/bias/Read/ReadVariableOpReadVariableOp(Adam/v/multi_head_attention_6/query/bias*
_output_shapes

:*
dtype0
�
(Adam/m/multi_head_attention_6/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(Adam/m/multi_head_attention_6/query/bias
�
<Adam/m/multi_head_attention_6/query/bias/Read/ReadVariableOpReadVariableOp(Adam/m/multi_head_attention_6/query/bias*
_output_shapes

:*
dtype0
�
*Adam/v/multi_head_attention_6/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/v/multi_head_attention_6/query/kernel
�
>Adam/v/multi_head_attention_6/query/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/multi_head_attention_6/query/kernel*"
_output_shapes
:*
dtype0
�
*Adam/m/multi_head_attention_6/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/m/multi_head_attention_6/query/kernel
�
>Adam/m/multi_head_attention_6/query/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/multi_head_attention_6/query/kernel*"
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
,multi_head_attention_7/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,multi_head_attention_7/attention_output/bias
�
@multi_head_attention_7/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_7/attention_output/bias*
_output_shapes
:*
dtype0
�
.multi_head_attention_7/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.multi_head_attention_7/attention_output/kernel
�
Bmulti_head_attention_7/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_7/attention_output/kernel*"
_output_shapes
:*
dtype0
�
!multi_head_attention_7/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!multi_head_attention_7/value/bias
�
5multi_head_attention_7/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_7/value/bias*
_output_shapes

:*
dtype0
�
#multi_head_attention_7/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#multi_head_attention_7/value/kernel
�
7multi_head_attention_7/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_7/value/kernel*"
_output_shapes
:*
dtype0
�
multi_head_attention_7/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!multi_head_attention_7/key/bias
�
3multi_head_attention_7/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_7/key/bias*
_output_shapes

:*
dtype0
�
!multi_head_attention_7/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!multi_head_attention_7/key/kernel
�
5multi_head_attention_7/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_7/key/kernel*"
_output_shapes
:*
dtype0
�
!multi_head_attention_7/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!multi_head_attention_7/query/bias
�
5multi_head_attention_7/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_7/query/bias*
_output_shapes

:*
dtype0
�
#multi_head_attention_7/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#multi_head_attention_7/query/kernel
�
7multi_head_attention_7/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_7/query/kernel*"
_output_shapes
:*
dtype0
�
,multi_head_attention_6/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,multi_head_attention_6/attention_output/bias
�
@multi_head_attention_6/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_6/attention_output/bias*
_output_shapes
:*
dtype0
�
.multi_head_attention_6/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.multi_head_attention_6/attention_output/kernel
�
Bmulti_head_attention_6/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_6/attention_output/kernel*"
_output_shapes
:*
dtype0
�
!multi_head_attention_6/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!multi_head_attention_6/value/bias
�
5multi_head_attention_6/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_6/value/bias*
_output_shapes

:*
dtype0
�
#multi_head_attention_6/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#multi_head_attention_6/value/kernel
�
7multi_head_attention_6/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_6/value/kernel*"
_output_shapes
:*
dtype0
�
multi_head_attention_6/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!multi_head_attention_6/key/bias
�
3multi_head_attention_6/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_6/key/bias*
_output_shapes

:*
dtype0
�
!multi_head_attention_6/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!multi_head_attention_6/key/kernel
�
5multi_head_attention_6/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_6/key/kernel*"
_output_shapes
:*
dtype0
�
!multi_head_attention_6/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!multi_head_attention_6/query/bias
�
5multi_head_attention_6/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_6/query/bias*
_output_shapes

:*
dtype0
�
#multi_head_attention_6/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#multi_head_attention_6/query/kernel
�
7multi_head_attention_6/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_6/query/kernel*"
_output_shapes
:*
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:*
dtype0
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:*
dtype0
�
layer_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_15/beta
�
/layer_normalization_15/beta/Read/ReadVariableOpReadVariableOplayer_normalization_15/beta*
_output_shapes
:*
dtype0
�
layer_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namelayer_normalization_15/gamma
�
0layer_normalization_15/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_15/gamma*
_output_shapes
:*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:*
dtype0
z
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_21/kernel
s
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes

:*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
z
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

:*
dtype0
�
layer_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_14/beta
�
/layer_normalization_14/beta/Read/ReadVariableOpReadVariableOplayer_normalization_14/beta*
_output_shapes
:*
dtype0
�
layer_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namelayer_normalization_14/gamma
�
0layer_normalization_14/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_14/gamma*
_output_shapes
:*
dtype0
�
layer_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_13/beta
�
/layer_normalization_13/beta/Read/ReadVariableOpReadVariableOplayer_normalization_13/beta*
_output_shapes
:*
dtype0
�
layer_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namelayer_normalization_13/gamma
�
0layer_normalization_13/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_13/gamma*
_output_shapes
:*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:*
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
z
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_18/kernel
s
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes

:*
dtype0
�
layer_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_12/beta
�
/layer_normalization_12/beta/Read/ReadVariableOpReadVariableOplayer_normalization_12/beta*
_output_shapes
:*
dtype0
�
layer_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namelayer_normalization_12/gamma
�
0layer_normalization_12/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_12/gamma*
_output_shapes
:*
dtype0
�
serving_default_input_7Placeholder*+
_output_shapes
:���������>*
dtype0* 
shape:���������>
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7Const#multi_head_attention_6/query/kernel!multi_head_attention_6/query/bias!multi_head_attention_6/key/kernelmulti_head_attention_6/key/bias#multi_head_attention_6/value/kernel!multi_head_attention_6/value/bias.multi_head_attention_6/attention_output/kernel,multi_head_attention_6/attention_output/biaslayer_normalization_12/gammalayer_normalization_12/betadense_18/kernel#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_19/kerneldense_19/biaslayer_normalization_13/gammalayer_normalization_13/beta#multi_head_attention_7/query/kernel!multi_head_attention_7/query/bias!multi_head_attention_7/key/kernelmulti_head_attention_7/key/bias#multi_head_attention_7/value/kernel!multi_head_attention_7/value/bias.multi_head_attention_7/attention_output/kernel,multi_head_attention_7/attention_output/biaslayer_normalization_14/gammalayer_normalization_14/betadense_20/kernel%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betadense_21/kerneldense_21/biaslayer_normalization_15/gammalayer_normalization_15/betadense_22/kerneldense_22/bias*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_516286

NoOpNoOp
��
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*ɡ
value��B�� B��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer-17
layer-18
layer_with_weights-10
layer-19
layer-20
layer_with_weights-11
layer-21
layer-22
layer_with_weights-12
layer-23
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
 	optimizer
!
signatures*
* 
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses* 
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
._query_dense
/
_key_dense
0_value_dense
1_softmax
2_dropout_layer
3_output_dense*
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses* 
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@axis
	Agamma
Bbeta*
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel*
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses* 
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_random_generator* 
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias*
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses* 
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses
vaxis
	wgamma
xbeta*
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�0
�1
�2
�3
�4
�5
�6
�7
A8
B9
I10
Q11
R12
S13
T14
h15
i16
w17
x18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39*
�
�0
�1
�2
�3
�4
�5
�6
�7
A8
B9
I10
Q11
R12
h13
i14
w15
x16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 

�	capture_0* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

A0
B1*

A0
B1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_12/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_12/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*

I0*

I0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
 
Q0
R1
S2
T3*

Q0
R1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

h0
i1*

h0
i1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_19/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

w0
x1*

w0
x1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_13/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_13/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_14/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_14/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_21/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_21/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
lf
VARIABLE_VALUElayer_normalization_15/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUElayer_normalization_15/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_22/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_22/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#multi_head_attention_6/query/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!multi_head_attention_6/query/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!multi_head_attention_6/key/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention_6/key/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#multi_head_attention_6/value/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!multi_head_attention_6/value/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.multi_head_attention_6/attention_output/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,multi_head_attention_6/attention_output/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_7/query/kernel'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_7/query/bias'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_7/key/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention_7/key/bias'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_7/value/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_7/value/bias'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.multi_head_attention_7/attention_output/kernel'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention_7/attention_output/bias'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
"
S0
T1
�2
�3*
�
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
10
11
12
13
14
15
16
17
18
19
20
21
22
23*

�0
�1*
* 
* 

�	capture_0* 

�	capture_0* 

�	capture_0* 

�	capture_0* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35*
* 

�	capture_0* 
* 
* 
* 
* 
* 

�	capture_0* 

�	capture_0* 
* 
.
.0
/1
02
13
24
35*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
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

S0
T1*
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
3
0
�1
�2
�3
�4
�5*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
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
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
uo
VARIABLE_VALUE*Adam/m/multi_head_attention_6/query/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/v/multi_head_attention_6/query/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/m/multi_head_attention_6/query/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/v/multi_head_attention_6/query/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/m/multi_head_attention_6/key/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/v/multi_head_attention_6/key/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/m/multi_head_attention_6/key/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/v/multi_head_attention_6/key/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/m/multi_head_attention_6/value/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/multi_head_attention_6/value/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention_6/value/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention_6/value/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE5Adam/m/multi_head_attention_6/attention_output/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE5Adam/v/multi_head_attention_6/attention_output/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/m/multi_head_attention_6/attention_output/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/v/multi_head_attention_6/attention_output/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/layer_normalization_12/gamma2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/layer_normalization_12/gamma2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_12/beta2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_12/beta2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_18/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_18/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/batch_normalization/gamma2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/batch_normalization/gamma2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/batch_normalization/beta2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/batch_normalization/beta2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_19/kernel2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_19/kernel2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_19/bias2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_19/bias2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/layer_normalization_13/gamma2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/layer_normalization_13/gamma2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_13/beta2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_13/beta2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/m/multi_head_attention_7/query/kernel2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/multi_head_attention_7/query/kernel2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention_7/query/bias2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention_7/query/bias2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention_7/key/kernel2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention_7/key/kernel2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/multi_head_attention_7/key/bias2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/multi_head_attention_7/key/bias2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/m/multi_head_attention_7/value/kernel2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/multi_head_attention_7/value/kernel2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/multi_head_attention_7/value/bias2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/multi_head_attention_7/value/bias2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE5Adam/m/multi_head_attention_7/attention_output/kernel2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE5Adam/v/multi_head_attention_7/attention_output/kernel2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/m/multi_head_attention_7/attention_output/bias2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE3Adam/v/multi_head_attention_7/attention_output/bias2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/layer_normalization_14/gamma2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/layer_normalization_14/gamma2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_14/beta2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_14/beta2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_20/kernel2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_20/kernel2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_1/gamma2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_1/gamma2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_1/beta2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_1/beta2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_21/kernel2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_21/kernel2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_21/bias2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_21/bias2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/layer_normalization_15/gamma2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/layer_normalization_15/gamma2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_15/beta2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_15/beta2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_22/kernel2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_22/kernel2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_22/bias2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_22/bias2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUE*
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

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�"
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamelayer_normalization_12/gammalayer_normalization_12/betadense_18/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense_19/kerneldense_19/biaslayer_normalization_13/gammalayer_normalization_13/betalayer_normalization_14/gammalayer_normalization_14/betadense_20/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense_21/kerneldense_21/biaslayer_normalization_15/gammalayer_normalization_15/betadense_22/kerneldense_22/bias#multi_head_attention_6/query/kernel!multi_head_attention_6/query/bias!multi_head_attention_6/key/kernelmulti_head_attention_6/key/bias#multi_head_attention_6/value/kernel!multi_head_attention_6/value/bias.multi_head_attention_6/attention_output/kernel,multi_head_attention_6/attention_output/bias#multi_head_attention_7/query/kernel!multi_head_attention_7/query/bias!multi_head_attention_7/key/kernelmulti_head_attention_7/key/bias#multi_head_attention_7/value/kernel!multi_head_attention_7/value/bias.multi_head_attention_7/attention_output/kernel,multi_head_attention_7/attention_output/bias	iterationlearning_rate*Adam/m/multi_head_attention_6/query/kernel*Adam/v/multi_head_attention_6/query/kernel(Adam/m/multi_head_attention_6/query/bias(Adam/v/multi_head_attention_6/query/bias(Adam/m/multi_head_attention_6/key/kernel(Adam/v/multi_head_attention_6/key/kernel&Adam/m/multi_head_attention_6/key/bias&Adam/v/multi_head_attention_6/key/bias*Adam/m/multi_head_attention_6/value/kernel*Adam/v/multi_head_attention_6/value/kernel(Adam/m/multi_head_attention_6/value/bias(Adam/v/multi_head_attention_6/value/bias5Adam/m/multi_head_attention_6/attention_output/kernel5Adam/v/multi_head_attention_6/attention_output/kernel3Adam/m/multi_head_attention_6/attention_output/bias3Adam/v/multi_head_attention_6/attention_output/bias#Adam/m/layer_normalization_12/gamma#Adam/v/layer_normalization_12/gamma"Adam/m/layer_normalization_12/beta"Adam/v/layer_normalization_12/betaAdam/m/dense_18/kernelAdam/v/dense_18/kernel Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/betaAdam/m/dense_19/kernelAdam/v/dense_19/kernelAdam/m/dense_19/biasAdam/v/dense_19/bias#Adam/m/layer_normalization_13/gamma#Adam/v/layer_normalization_13/gamma"Adam/m/layer_normalization_13/beta"Adam/v/layer_normalization_13/beta*Adam/m/multi_head_attention_7/query/kernel*Adam/v/multi_head_attention_7/query/kernel(Adam/m/multi_head_attention_7/query/bias(Adam/v/multi_head_attention_7/query/bias(Adam/m/multi_head_attention_7/key/kernel(Adam/v/multi_head_attention_7/key/kernel&Adam/m/multi_head_attention_7/key/bias&Adam/v/multi_head_attention_7/key/bias*Adam/m/multi_head_attention_7/value/kernel*Adam/v/multi_head_attention_7/value/kernel(Adam/m/multi_head_attention_7/value/bias(Adam/v/multi_head_attention_7/value/bias5Adam/m/multi_head_attention_7/attention_output/kernel5Adam/v/multi_head_attention_7/attention_output/kernel3Adam/m/multi_head_attention_7/attention_output/bias3Adam/v/multi_head_attention_7/attention_output/bias#Adam/m/layer_normalization_14/gamma#Adam/v/layer_normalization_14/gamma"Adam/m/layer_normalization_14/beta"Adam/v/layer_normalization_14/betaAdam/m/dense_20/kernelAdam/v/dense_20/kernel"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/betaAdam/m/dense_21/kernelAdam/v/dense_21/kernelAdam/m/dense_21/biasAdam/v/dense_21/bias#Adam/m/layer_normalization_15/gamma#Adam/v/layer_normalization_15/gamma"Adam/m/layer_normalization_15/beta"Adam/v/layer_normalization_15/betaAdam/m/dense_22/kernelAdam/v/dense_22/kernelAdam/m/dense_22/biasAdam/v/dense_22/biastotal_1count_1totalcountConst_1*�
Tin|
z2x*
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
GPU 2J 8� *(
f#R!
__inference__traced_save_517873
�"
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization_12/gammalayer_normalization_12/betadense_18/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense_19/kerneldense_19/biaslayer_normalization_13/gammalayer_normalization_13/betalayer_normalization_14/gammalayer_normalization_14/betadense_20/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense_21/kerneldense_21/biaslayer_normalization_15/gammalayer_normalization_15/betadense_22/kerneldense_22/bias#multi_head_attention_6/query/kernel!multi_head_attention_6/query/bias!multi_head_attention_6/key/kernelmulti_head_attention_6/key/bias#multi_head_attention_6/value/kernel!multi_head_attention_6/value/bias.multi_head_attention_6/attention_output/kernel,multi_head_attention_6/attention_output/bias#multi_head_attention_7/query/kernel!multi_head_attention_7/query/bias!multi_head_attention_7/key/kernelmulti_head_attention_7/key/bias#multi_head_attention_7/value/kernel!multi_head_attention_7/value/bias.multi_head_attention_7/attention_output/kernel,multi_head_attention_7/attention_output/bias	iterationlearning_rate*Adam/m/multi_head_attention_6/query/kernel*Adam/v/multi_head_attention_6/query/kernel(Adam/m/multi_head_attention_6/query/bias(Adam/v/multi_head_attention_6/query/bias(Adam/m/multi_head_attention_6/key/kernel(Adam/v/multi_head_attention_6/key/kernel&Adam/m/multi_head_attention_6/key/bias&Adam/v/multi_head_attention_6/key/bias*Adam/m/multi_head_attention_6/value/kernel*Adam/v/multi_head_attention_6/value/kernel(Adam/m/multi_head_attention_6/value/bias(Adam/v/multi_head_attention_6/value/bias5Adam/m/multi_head_attention_6/attention_output/kernel5Adam/v/multi_head_attention_6/attention_output/kernel3Adam/m/multi_head_attention_6/attention_output/bias3Adam/v/multi_head_attention_6/attention_output/bias#Adam/m/layer_normalization_12/gamma#Adam/v/layer_normalization_12/gamma"Adam/m/layer_normalization_12/beta"Adam/v/layer_normalization_12/betaAdam/m/dense_18/kernelAdam/v/dense_18/kernel Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/betaAdam/m/dense_19/kernelAdam/v/dense_19/kernelAdam/m/dense_19/biasAdam/v/dense_19/bias#Adam/m/layer_normalization_13/gamma#Adam/v/layer_normalization_13/gamma"Adam/m/layer_normalization_13/beta"Adam/v/layer_normalization_13/beta*Adam/m/multi_head_attention_7/query/kernel*Adam/v/multi_head_attention_7/query/kernel(Adam/m/multi_head_attention_7/query/bias(Adam/v/multi_head_attention_7/query/bias(Adam/m/multi_head_attention_7/key/kernel(Adam/v/multi_head_attention_7/key/kernel&Adam/m/multi_head_attention_7/key/bias&Adam/v/multi_head_attention_7/key/bias*Adam/m/multi_head_attention_7/value/kernel*Adam/v/multi_head_attention_7/value/kernel(Adam/m/multi_head_attention_7/value/bias(Adam/v/multi_head_attention_7/value/bias5Adam/m/multi_head_attention_7/attention_output/kernel5Adam/v/multi_head_attention_7/attention_output/kernel3Adam/m/multi_head_attention_7/attention_output/bias3Adam/v/multi_head_attention_7/attention_output/bias#Adam/m/layer_normalization_14/gamma#Adam/v/layer_normalization_14/gamma"Adam/m/layer_normalization_14/beta"Adam/v/layer_normalization_14/betaAdam/m/dense_20/kernelAdam/v/dense_20/kernel"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/betaAdam/m/dense_21/kernelAdam/v/dense_21/kernelAdam/m/dense_21/biasAdam/v/dense_21/bias#Adam/m/layer_normalization_15/gamma#Adam/v/layer_normalization_15/gamma"Adam/m/layer_normalization_15/beta"Adam/v/layer_normalization_15/betaAdam/m/dense_22/kernelAdam/v/dense_22/kernelAdam/m/dense_22/biasAdam/v/dense_22/biastotal_1count_1totalcount*�
Tin{
y2w*
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
GPU 2J 8� *+
f&R$
"__inference__traced_restore_518236��
�!
�	
(__inference_model_6_layer_call_fn_515883
input_7
unknown
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22: 

unknown_23:

unknown_24: 

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*F
_read_only_resource_inputs(
&$	
"#$%&'()*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_515607o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesw
u:���������>:>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&)"
 
_user_specified_name515879:&("
 
_user_specified_name515877:&'"
 
_user_specified_name515875:&&"
 
_user_specified_name515873:&%"
 
_user_specified_name515871:&$"
 
_user_specified_name515869:&#"
 
_user_specified_name515867:&""
 
_user_specified_name515865:&!"
 
_user_specified_name515863:& "
 
_user_specified_name515861:&"
 
_user_specified_name515859:&"
 
_user_specified_name515857:&"
 
_user_specified_name515855:&"
 
_user_specified_name515853:&"
 
_user_specified_name515851:&"
 
_user_specified_name515849:&"
 
_user_specified_name515847:&"
 
_user_specified_name515845:&"
 
_user_specified_name515843:&"
 
_user_specified_name515841:&"
 
_user_specified_name515839:&"
 
_user_specified_name515837:&"
 
_user_specified_name515835:&"
 
_user_specified_name515833:&"
 
_user_specified_name515831:&"
 
_user_specified_name515829:&"
 
_user_specified_name515827:&"
 
_user_specified_name515825:&"
 
_user_specified_name515823:&"
 
_user_specified_name515821:&"
 
_user_specified_name515819:&
"
 
_user_specified_name515817:&	"
 
_user_specified_name515815:&"
 
_user_specified_name515813:&"
 
_user_specified_name515811:&"
 
_user_specified_name515809:&"
 
_user_specified_name515807:&"
 
_user_specified_name515805:&"
 
_user_specified_name515803:&"
 
_user_specified_name515801:JF
"
_output_shapes
:>
 
_user_specified_name515799:T P
+
_output_shapes
:���������>
!
_user_specified_name	input_7
�
F
*__inference_dropout_1_layer_call_fn_517012

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_515776d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
}
)__inference_dense_18_layer_call_fn_516488

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_515268s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������>: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name516484:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
l
B__inference_add_14_layer_call_and_return_conditional_losses_515433

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������>S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������>:���������>:SO
+
_output_shapes
:���������>
 
_user_specified_nameinputs:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�	
�
4__inference_batch_normalization_layer_call_fn_516528

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_514982|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name516524:&"
 
_user_specified_name516522:&"
 
_user_specified_name516520:&"
 
_user_specified_name516518:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�4
�
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_516403	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:͓>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������>�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������>>*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������>>Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������>>l
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
::���
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������>>*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������>>\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������>>�
einsum_1/EinsumEinsum!dropout/dropout/SelectV2:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������>*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������>*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:���������>�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������>:���������>: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������>

_user_specified_namevalue:R N
+
_output_shapes
:���������>

_user_specified_namequery
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_516632

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������>_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������>"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
D
(__inference_dropout_layer_call_fn_516615

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_515688d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_516992

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�,
�
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_516438	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:͓>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������>�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������>>*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������>>q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������>>�
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������>*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������>*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:���������>�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������>:���������>: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������>

_user_specified_namevalue:R N
+
_output_shapes
:���������>

_user_specified_namequery
�
d
H__inference_activation_1_layer_call_and_return_conditional_losses_517002

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:���������>^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
7__inference_layer_normalization_15_layer_call_fn_517089

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_515583s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name517085:&"
 
_user_specified_name517083:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
l
B__inference_add_15_layer_call_and_return_conditional_losses_515560

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������>S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������>:���������>:SO
+
_output_shapes
:���������>
 
_user_specified_nameinputs:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_515518

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������>Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������>*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������>T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:���������>e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
a
(__inference_dropout_layer_call_fn_516610

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_515298s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�,
�
R__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_515735	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:͓>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������>�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������>>*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������>>q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������>>�
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������>*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������>*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:���������>�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������>:���������>: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������>

_user_specified_namevalue:R N
+
_output_shapes
:���������>

_user_specified_namequery
�
�
7__inference_multi_head_attention_7_layer_call_fn_516758	
query	
value
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_515735s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������>:���������>: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&	"
 
_user_specified_name516754:&"
 
_user_specified_name516752:&"
 
_user_specified_name516750:&"
 
_user_specified_name516748:&"
 
_user_specified_name516746:&"
 
_user_specified_name516744:&"
 
_user_specified_name516742:&"
 
_user_specified_name516740:RN
+
_output_shapes
:���������>

_user_specified_namevalue:R N
+
_output_shapes
:���������>

_user_specified_namequery
�
z
Q__inference_positional_encoding_6_layer_call_and_return_conditional_losses_516317

inputs
unknown
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
ConstConst*
_output_shapes
: *
dtype0*
value	B : I
Const_1Const*
_output_shapes
: *
dtype0*
value	B :Y
strided_slice_1/stack/0Const*
_output_shapes
: *
dtype0*
value	B : Y
strided_slice_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : �
strided_slice_1/stackPack strided_slice_1/stack/0:output:0Const:output:0 strided_slice_1/stack/2:output:0*
N*
T0*
_output_shapes
:[
strided_slice_1/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : [
strided_slice_1/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : �
strided_slice_1/stack_1Pack"strided_slice_1/stack_1/0:output:0strided_slice:output:0"strided_slice_1/stack_1/2:output:0*
N*
T0*
_output_shapes
:[
strided_slice_1/stack_2/0Const*
_output_shapes
: *
dtype0*
value	B :[
strided_slice_1/stack_2/2Const*
_output_shapes
: *
dtype0*
value	B :�
strided_slice_1/stack_2Pack"strided_slice_1/stack_2/0:output:0Const_1:output:0"strided_slice_1/stack_2/2:output:0*
N*
T0*
_output_shapes
:�
strided_slice_1StridedSliceunknownstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*"
_output_shapes
:>*

begin_mask*
end_maskd
addAddV2inputsstrided_slice_1:output:0*
T0*+
_output_shapes
:���������>S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������>:>:JF
"
_output_shapes
:>
 
_user_specified_name516301:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
r
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_517122

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
D__inference_dense_19_layer_call_and_return_conditional_losses_516671

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������>�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������>r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������>V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
��
�,
!__inference__wrapped_model_514948
input_7(
$model_6_positional_encoding_6_514653`
Jmodel_6_multi_head_attention_6_query_einsum_einsum_readvariableop_resource:R
@model_6_multi_head_attention_6_query_add_readvariableop_resource:^
Hmodel_6_multi_head_attention_6_key_einsum_einsum_readvariableop_resource:P
>model_6_multi_head_attention_6_key_add_readvariableop_resource:`
Jmodel_6_multi_head_attention_6_value_einsum_einsum_readvariableop_resource:R
@model_6_multi_head_attention_6_value_add_readvariableop_resource:k
Umodel_6_multi_head_attention_6_attention_output_einsum_einsum_readvariableop_resource:Y
Kmodel_6_multi_head_attention_6_attention_output_add_readvariableop_resource:R
Dmodel_6_layer_normalization_12_batchnorm_mul_readvariableop_resource:N
@model_6_layer_normalization_12_batchnorm_readvariableop_resource:D
2model_6_dense_18_tensordot_readvariableop_resource:K
=model_6_batch_normalization_batchnorm_readvariableop_resource:O
Amodel_6_batch_normalization_batchnorm_mul_readvariableop_resource:M
?model_6_batch_normalization_batchnorm_readvariableop_1_resource:M
?model_6_batch_normalization_batchnorm_readvariableop_2_resource:D
2model_6_dense_19_tensordot_readvariableop_resource:>
0model_6_dense_19_biasadd_readvariableop_resource:R
Dmodel_6_layer_normalization_13_batchnorm_mul_readvariableop_resource:N
@model_6_layer_normalization_13_batchnorm_readvariableop_resource:`
Jmodel_6_multi_head_attention_7_query_einsum_einsum_readvariableop_resource:R
@model_6_multi_head_attention_7_query_add_readvariableop_resource:^
Hmodel_6_multi_head_attention_7_key_einsum_einsum_readvariableop_resource:P
>model_6_multi_head_attention_7_key_add_readvariableop_resource:`
Jmodel_6_multi_head_attention_7_value_einsum_einsum_readvariableop_resource:R
@model_6_multi_head_attention_7_value_add_readvariableop_resource:k
Umodel_6_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource:Y
Kmodel_6_multi_head_attention_7_attention_output_add_readvariableop_resource:R
Dmodel_6_layer_normalization_14_batchnorm_mul_readvariableop_resource:N
@model_6_layer_normalization_14_batchnorm_readvariableop_resource:D
2model_6_dense_20_tensordot_readvariableop_resource:M
?model_6_batch_normalization_1_batchnorm_readvariableop_resource:Q
Cmodel_6_batch_normalization_1_batchnorm_mul_readvariableop_resource:O
Amodel_6_batch_normalization_1_batchnorm_readvariableop_1_resource:O
Amodel_6_batch_normalization_1_batchnorm_readvariableop_2_resource:D
2model_6_dense_21_tensordot_readvariableop_resource:>
0model_6_dense_21_biasadd_readvariableop_resource:R
Dmodel_6_layer_normalization_15_batchnorm_mul_readvariableop_resource:N
@model_6_layer_normalization_15_batchnorm_readvariableop_resource:A
/model_6_dense_22_matmul_readvariableop_resource:>
0model_6_dense_22_biasadd_readvariableop_resource:
identity��4model_6/batch_normalization/batchnorm/ReadVariableOp�6model_6/batch_normalization/batchnorm/ReadVariableOp_1�6model_6/batch_normalization/batchnorm/ReadVariableOp_2�8model_6/batch_normalization/batchnorm/mul/ReadVariableOp�6model_6/batch_normalization_1/batchnorm/ReadVariableOp�8model_6/batch_normalization_1/batchnorm/ReadVariableOp_1�8model_6/batch_normalization_1/batchnorm/ReadVariableOp_2�:model_6/batch_normalization_1/batchnorm/mul/ReadVariableOp�)model_6/dense_18/Tensordot/ReadVariableOp�'model_6/dense_19/BiasAdd/ReadVariableOp�)model_6/dense_19/Tensordot/ReadVariableOp�)model_6/dense_20/Tensordot/ReadVariableOp�'model_6/dense_21/BiasAdd/ReadVariableOp�)model_6/dense_21/Tensordot/ReadVariableOp�'model_6/dense_22/BiasAdd/ReadVariableOp�&model_6/dense_22/MatMul/ReadVariableOp�7model_6/layer_normalization_12/batchnorm/ReadVariableOp�;model_6/layer_normalization_12/batchnorm/mul/ReadVariableOp�7model_6/layer_normalization_13/batchnorm/ReadVariableOp�;model_6/layer_normalization_13/batchnorm/mul/ReadVariableOp�7model_6/layer_normalization_14/batchnorm/ReadVariableOp�;model_6/layer_normalization_14/batchnorm/mul/ReadVariableOp�7model_6/layer_normalization_15/batchnorm/ReadVariableOp�;model_6/layer_normalization_15/batchnorm/mul/ReadVariableOp�Bmodel_6/multi_head_attention_6/attention_output/add/ReadVariableOp�Lmodel_6/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp�5model_6/multi_head_attention_6/key/add/ReadVariableOp�?model_6/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp�7model_6/multi_head_attention_6/query/add/ReadVariableOp�Amodel_6/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp�7model_6/multi_head_attention_6/value/add/ReadVariableOp�Amodel_6/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp�Bmodel_6/multi_head_attention_7/attention_output/add/ReadVariableOp�Lmodel_6/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp�5model_6/multi_head_attention_7/key/add/ReadVariableOp�?model_6/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp�7model_6/multi_head_attention_7/query/add/ReadVariableOp�Amodel_6/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp�7model_6/multi_head_attention_7/value/add/ReadVariableOp�Amodel_6/multi_head_attention_7/value/einsum/Einsum/ReadVariableOph
#model_6/positional_encoding_6/ShapeShapeinput_7*
T0*
_output_shapes
::��{
1model_6/positional_encoding_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model_6/positional_encoding_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_6/positional_encoding_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+model_6/positional_encoding_6/strided_sliceStridedSlice,model_6/positional_encoding_6/Shape:output:0:model_6/positional_encoding_6/strided_slice/stack:output:0<model_6/positional_encoding_6/strided_slice/stack_1:output:0<model_6/positional_encoding_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_6/positional_encoding_6/ConstConst*
_output_shapes
: *
dtype0*
value	B : g
%model_6/positional_encoding_6/Const_1Const*
_output_shapes
: *
dtype0*
value	B :w
5model_6/positional_encoding_6/strided_slice_1/stack/0Const*
_output_shapes
: *
dtype0*
value	B : w
5model_6/positional_encoding_6/strided_slice_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : �
3model_6/positional_encoding_6/strided_slice_1/stackPack>model_6/positional_encoding_6/strided_slice_1/stack/0:output:0,model_6/positional_encoding_6/Const:output:0>model_6/positional_encoding_6/strided_slice_1/stack/2:output:0*
N*
T0*
_output_shapes
:y
7model_6/positional_encoding_6/strided_slice_1/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : y
7model_6/positional_encoding_6/strided_slice_1/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : �
5model_6/positional_encoding_6/strided_slice_1/stack_1Pack@model_6/positional_encoding_6/strided_slice_1/stack_1/0:output:04model_6/positional_encoding_6/strided_slice:output:0@model_6/positional_encoding_6/strided_slice_1/stack_1/2:output:0*
N*
T0*
_output_shapes
:y
7model_6/positional_encoding_6/strided_slice_1/stack_2/0Const*
_output_shapes
: *
dtype0*
value	B :y
7model_6/positional_encoding_6/strided_slice_1/stack_2/2Const*
_output_shapes
: *
dtype0*
value	B :�
5model_6/positional_encoding_6/strided_slice_1/stack_2Pack@model_6/positional_encoding_6/strided_slice_1/stack_2/0:output:0.model_6/positional_encoding_6/Const_1:output:0@model_6/positional_encoding_6/strided_slice_1/stack_2/2:output:0*
N*
T0*
_output_shapes
:�
-model_6/positional_encoding_6/strided_slice_1StridedSlice$model_6_positional_encoding_6_514653<model_6/positional_encoding_6/strided_slice_1/stack:output:0>model_6/positional_encoding_6/strided_slice_1/stack_1:output:0>model_6/positional_encoding_6/strided_slice_1/stack_2:output:0*
Index0*
T0*"
_output_shapes
:>*

begin_mask*
end_mask�
!model_6/positional_encoding_6/addAddV2input_76model_6/positional_encoding_6/strided_slice_1:output:0*
T0*+
_output_shapes
:���������>�
Amodel_6/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpReadVariableOpJmodel_6_multi_head_attention_6_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
2model_6/multi_head_attention_6/query/einsum/EinsumEinsum%model_6/positional_encoding_6/add:z:0Imodel_6/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abde�
7model_6/multi_head_attention_6/query/add/ReadVariableOpReadVariableOp@model_6_multi_head_attention_6_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
(model_6/multi_head_attention_6/query/addAddV2;model_6/multi_head_attention_6/query/einsum/Einsum:output:0?model_6/multi_head_attention_6/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
?model_6/multi_head_attention_6/key/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_6_multi_head_attention_6_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
0model_6/multi_head_attention_6/key/einsum/EinsumEinsum%model_6/positional_encoding_6/add:z:0Gmodel_6/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abde�
5model_6/multi_head_attention_6/key/add/ReadVariableOpReadVariableOp>model_6_multi_head_attention_6_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&model_6/multi_head_attention_6/key/addAddV29model_6/multi_head_attention_6/key/einsum/Einsum:output:0=model_6/multi_head_attention_6/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
Amodel_6/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpReadVariableOpJmodel_6_multi_head_attention_6_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
2model_6/multi_head_attention_6/value/einsum/EinsumEinsum%model_6/positional_encoding_6/add:z:0Imodel_6/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abde�
7model_6/multi_head_attention_6/value/add/ReadVariableOpReadVariableOp@model_6_multi_head_attention_6_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
(model_6/multi_head_attention_6/value/addAddV2;model_6/multi_head_attention_6/value/einsum/Einsum:output:0?model_6/multi_head_attention_6/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>i
$model_6/multi_head_attention_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:͓>�
"model_6/multi_head_attention_6/MulMul,model_6/multi_head_attention_6/query/add:z:0-model_6/multi_head_attention_6/Mul/y:output:0*
T0*/
_output_shapes
:���������>�
,model_6/multi_head_attention_6/einsum/EinsumEinsum*model_6/multi_head_attention_6/key/add:z:0&model_6/multi_head_attention_6/Mul:z:0*
N*
T0*/
_output_shapes
:���������>>*
equationaecd,abcd->acbe�
.model_6/multi_head_attention_6/softmax/SoftmaxSoftmax5model_6/multi_head_attention_6/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������>>�
/model_6/multi_head_attention_6/dropout/IdentityIdentity8model_6/multi_head_attention_6/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������>>�
.model_6/multi_head_attention_6/einsum_1/EinsumEinsum8model_6/multi_head_attention_6/dropout/Identity:output:0,model_6/multi_head_attention_6/value/add:z:0*
N*
T0*/
_output_shapes
:���������>*
equationacbe,aecd->abcd�
Lmodel_6/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpUmodel_6_multi_head_attention_6_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
=model_6/multi_head_attention_6/attention_output/einsum/EinsumEinsum7model_6/multi_head_attention_6/einsum_1/Einsum:output:0Tmodel_6/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������>*
equationabcd,cde->abe�
Bmodel_6/multi_head_attention_6/attention_output/add/ReadVariableOpReadVariableOpKmodel_6_multi_head_attention_6_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
3model_6/multi_head_attention_6/attention_output/addAddV2Fmodel_6/multi_head_attention_6/attention_output/einsum/Einsum:output:0Jmodel_6/multi_head_attention_6/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>�
model_6/add_12/addAddV2%model_6/positional_encoding_6/add:z:07model_6/multi_head_attention_6/attention_output/add:z:0*
T0*+
_output_shapes
:���������>�
=model_6/layer_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
+model_6/layer_normalization_12/moments/meanMeanmodel_6/add_12/add:z:0Fmodel_6/layer_normalization_12/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(�
3model_6/layer_normalization_12/moments/StopGradientStopGradient4model_6/layer_normalization_12/moments/mean:output:0*
T0*+
_output_shapes
:���������>�
8model_6/layer_normalization_12/moments/SquaredDifferenceSquaredDifferencemodel_6/add_12/add:z:0<model_6/layer_normalization_12/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������>�
Amodel_6/layer_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
/model_6/layer_normalization_12/moments/varianceMean<model_6/layer_normalization_12/moments/SquaredDifference:z:0Jmodel_6/layer_normalization_12/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(s
.model_6/layer_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
,model_6/layer_normalization_12/batchnorm/addAddV28model_6/layer_normalization_12/moments/variance:output:07model_6/layer_normalization_12/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_12/batchnorm/RsqrtRsqrt0model_6/layer_normalization_12/batchnorm/add:z:0*
T0*+
_output_shapes
:���������>�
;model_6/layer_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_6_layer_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
,model_6/layer_normalization_12/batchnorm/mulMul2model_6/layer_normalization_12/batchnorm/Rsqrt:y:0Cmodel_6/layer_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_12/batchnorm/mul_1Mulmodel_6/add_12/add:z:00model_6/layer_normalization_12/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_12/batchnorm/mul_2Mul4model_6/layer_normalization_12/moments/mean:output:00model_6/layer_normalization_12/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>�
7model_6/layer_normalization_12/batchnorm/ReadVariableOpReadVariableOp@model_6_layer_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
,model_6/layer_normalization_12/batchnorm/subSub?model_6/layer_normalization_12/batchnorm/ReadVariableOp:value:02model_6/layer_normalization_12/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_12/batchnorm/add_1AddV22model_6/layer_normalization_12/batchnorm/mul_1:z:00model_6/layer_normalization_12/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������>�
)model_6/dense_18/Tensordot/ReadVariableOpReadVariableOp2model_6_dense_18_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0i
model_6/dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
model_6/dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
 model_6/dense_18/Tensordot/ShapeShape2model_6/layer_normalization_12/batchnorm/add_1:z:0*
T0*
_output_shapes
::��j
(model_6/dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#model_6/dense_18/Tensordot/GatherV2GatherV2)model_6/dense_18/Tensordot/Shape:output:0(model_6/dense_18/Tensordot/free:output:01model_6/dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*model_6/dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%model_6/dense_18/Tensordot/GatherV2_1GatherV2)model_6/dense_18/Tensordot/Shape:output:0(model_6/dense_18/Tensordot/axes:output:03model_6/dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 model_6/dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model_6/dense_18/Tensordot/ProdProd,model_6/dense_18/Tensordot/GatherV2:output:0)model_6/dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"model_6/dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!model_6/dense_18/Tensordot/Prod_1Prod.model_6/dense_18/Tensordot/GatherV2_1:output:0+model_6/dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&model_6/dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!model_6/dense_18/Tensordot/concatConcatV2(model_6/dense_18/Tensordot/free:output:0(model_6/dense_18/Tensordot/axes:output:0/model_6/dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 model_6/dense_18/Tensordot/stackPack(model_6/dense_18/Tensordot/Prod:output:0*model_6/dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$model_6/dense_18/Tensordot/transpose	Transpose2model_6/layer_normalization_12/batchnorm/add_1:z:0*model_6/dense_18/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������>�
"model_6/dense_18/Tensordot/ReshapeReshape(model_6/dense_18/Tensordot/transpose:y:0)model_6/dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!model_6/dense_18/Tensordot/MatMulMatMul+model_6/dense_18/Tensordot/Reshape:output:01model_6/dense_18/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
"model_6/dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(model_6/dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#model_6/dense_18/Tensordot/concat_1ConcatV2,model_6/dense_18/Tensordot/GatherV2:output:0+model_6/dense_18/Tensordot/Const_2:output:01model_6/dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model_6/dense_18/TensordotReshape+model_6/dense_18/Tensordot/MatMul:product:0,model_6/dense_18/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������>�
4model_6/batch_normalization/batchnorm/ReadVariableOpReadVariableOp=model_6_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
+model_6/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
)model_6/batch_normalization/batchnorm/addAddV2<model_6/batch_normalization/batchnorm/ReadVariableOp:value:04model_6/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
+model_6/batch_normalization/batchnorm/RsqrtRsqrt-model_6/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:�
8model_6/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_6_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
)model_6/batch_normalization/batchnorm/mulMul/model_6/batch_normalization/batchnorm/Rsqrt:y:0@model_6/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
+model_6/batch_normalization/batchnorm/mul_1Mul#model_6/dense_18/Tensordot:output:0-model_6/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>�
6model_6/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp?model_6_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
+model_6/batch_normalization/batchnorm/mul_2Mul>model_6/batch_normalization/batchnorm/ReadVariableOp_1:value:0-model_6/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:�
6model_6/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp?model_6_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
)model_6/batch_normalization/batchnorm/subSub>model_6/batch_normalization/batchnorm/ReadVariableOp_2:value:0/model_6/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
+model_6/batch_normalization/batchnorm/add_1AddV2/model_6/batch_normalization/batchnorm/mul_1:z:0-model_6/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������>�
model_6/activation/ReluRelu/model_6/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������>�
model_6/dropout/IdentityIdentity%model_6/activation/Relu:activations:0*
T0*+
_output_shapes
:���������>�
)model_6/dense_19/Tensordot/ReadVariableOpReadVariableOp2model_6_dense_19_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0i
model_6/dense_19/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
model_6/dense_19/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
 model_6/dense_19/Tensordot/ShapeShape!model_6/dropout/Identity:output:0*
T0*
_output_shapes
::��j
(model_6/dense_19/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#model_6/dense_19/Tensordot/GatherV2GatherV2)model_6/dense_19/Tensordot/Shape:output:0(model_6/dense_19/Tensordot/free:output:01model_6/dense_19/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*model_6/dense_19/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%model_6/dense_19/Tensordot/GatherV2_1GatherV2)model_6/dense_19/Tensordot/Shape:output:0(model_6/dense_19/Tensordot/axes:output:03model_6/dense_19/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 model_6/dense_19/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model_6/dense_19/Tensordot/ProdProd,model_6/dense_19/Tensordot/GatherV2:output:0)model_6/dense_19/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"model_6/dense_19/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!model_6/dense_19/Tensordot/Prod_1Prod.model_6/dense_19/Tensordot/GatherV2_1:output:0+model_6/dense_19/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&model_6/dense_19/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!model_6/dense_19/Tensordot/concatConcatV2(model_6/dense_19/Tensordot/free:output:0(model_6/dense_19/Tensordot/axes:output:0/model_6/dense_19/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 model_6/dense_19/Tensordot/stackPack(model_6/dense_19/Tensordot/Prod:output:0*model_6/dense_19/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$model_6/dense_19/Tensordot/transpose	Transpose!model_6/dropout/Identity:output:0*model_6/dense_19/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������>�
"model_6/dense_19/Tensordot/ReshapeReshape(model_6/dense_19/Tensordot/transpose:y:0)model_6/dense_19/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!model_6/dense_19/Tensordot/MatMulMatMul+model_6/dense_19/Tensordot/Reshape:output:01model_6/dense_19/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
"model_6/dense_19/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(model_6/dense_19/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#model_6/dense_19/Tensordot/concat_1ConcatV2,model_6/dense_19/Tensordot/GatherV2:output:0+model_6/dense_19/Tensordot/Const_2:output:01model_6/dense_19/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model_6/dense_19/TensordotReshape+model_6/dense_19/Tensordot/MatMul:product:0,model_6/dense_19/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������>�
'model_6/dense_19/BiasAdd/ReadVariableOpReadVariableOp0model_6_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_6/dense_19/BiasAddBiasAdd#model_6/dense_19/Tensordot:output:0/model_6/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>�
model_6/add_13/addAddV22model_6/layer_normalization_12/batchnorm/add_1:z:0!model_6/dense_19/BiasAdd:output:0*
T0*+
_output_shapes
:���������>�
=model_6/layer_normalization_13/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
+model_6/layer_normalization_13/moments/meanMeanmodel_6/add_13/add:z:0Fmodel_6/layer_normalization_13/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(�
3model_6/layer_normalization_13/moments/StopGradientStopGradient4model_6/layer_normalization_13/moments/mean:output:0*
T0*+
_output_shapes
:���������>�
8model_6/layer_normalization_13/moments/SquaredDifferenceSquaredDifferencemodel_6/add_13/add:z:0<model_6/layer_normalization_13/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������>�
Amodel_6/layer_normalization_13/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
/model_6/layer_normalization_13/moments/varianceMean<model_6/layer_normalization_13/moments/SquaredDifference:z:0Jmodel_6/layer_normalization_13/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(s
.model_6/layer_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
,model_6/layer_normalization_13/batchnorm/addAddV28model_6/layer_normalization_13/moments/variance:output:07model_6/layer_normalization_13/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_13/batchnorm/RsqrtRsqrt0model_6/layer_normalization_13/batchnorm/add:z:0*
T0*+
_output_shapes
:���������>�
;model_6/layer_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_6_layer_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
,model_6/layer_normalization_13/batchnorm/mulMul2model_6/layer_normalization_13/batchnorm/Rsqrt:y:0Cmodel_6/layer_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_13/batchnorm/mul_1Mulmodel_6/add_13/add:z:00model_6/layer_normalization_13/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_13/batchnorm/mul_2Mul4model_6/layer_normalization_13/moments/mean:output:00model_6/layer_normalization_13/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>�
7model_6/layer_normalization_13/batchnorm/ReadVariableOpReadVariableOp@model_6_layer_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
,model_6/layer_normalization_13/batchnorm/subSub?model_6/layer_normalization_13/batchnorm/ReadVariableOp:value:02model_6/layer_normalization_13/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_13/batchnorm/add_1AddV22model_6/layer_normalization_13/batchnorm/mul_1:z:00model_6/layer_normalization_13/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������>�
Amodel_6/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpJmodel_6_multi_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
2model_6/multi_head_attention_7/query/einsum/EinsumEinsum2model_6/layer_normalization_13/batchnorm/add_1:z:0Imodel_6/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abde�
7model_6/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp@model_6_multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
(model_6/multi_head_attention_7/query/addAddV2;model_6/multi_head_attention_7/query/einsum/Einsum:output:0?model_6/multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
?model_6/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_6_multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
0model_6/multi_head_attention_7/key/einsum/EinsumEinsum2model_6/layer_normalization_13/batchnorm/add_1:z:0Gmodel_6/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abde�
5model_6/multi_head_attention_7/key/add/ReadVariableOpReadVariableOp>model_6_multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

:*
dtype0�
&model_6/multi_head_attention_7/key/addAddV29model_6/multi_head_attention_7/key/einsum/Einsum:output:0=model_6/multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
Amodel_6/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpJmodel_6_multi_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
2model_6/multi_head_attention_7/value/einsum/EinsumEinsum2model_6/layer_normalization_13/batchnorm/add_1:z:0Imodel_6/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abde�
7model_6/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp@model_6_multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
(model_6/multi_head_attention_7/value/addAddV2;model_6/multi_head_attention_7/value/einsum/Einsum:output:0?model_6/multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>i
$model_6/multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:͓>�
"model_6/multi_head_attention_7/MulMul,model_6/multi_head_attention_7/query/add:z:0-model_6/multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:���������>�
,model_6/multi_head_attention_7/einsum/EinsumEinsum*model_6/multi_head_attention_7/key/add:z:0&model_6/multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:���������>>*
equationaecd,abcd->acbe�
.model_6/multi_head_attention_7/softmax/SoftmaxSoftmax5model_6/multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������>>�
/model_6/multi_head_attention_7/dropout/IdentityIdentity8model_6/multi_head_attention_7/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������>>�
.model_6/multi_head_attention_7/einsum_1/EinsumEinsum8model_6/multi_head_attention_7/dropout/Identity:output:0,model_6/multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:���������>*
equationacbe,aecd->abcd�
Lmodel_6/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpUmodel_6_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
=model_6/multi_head_attention_7/attention_output/einsum/EinsumEinsum7model_6/multi_head_attention_7/einsum_1/Einsum:output:0Tmodel_6/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������>*
equationabcd,cde->abe�
Bmodel_6/multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpKmodel_6_multi_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
3model_6/multi_head_attention_7/attention_output/addAddV2Fmodel_6/multi_head_attention_7/attention_output/einsum/Einsum:output:0Jmodel_6/multi_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>�
model_6/add_14/addAddV22model_6/layer_normalization_13/batchnorm/add_1:z:07model_6/multi_head_attention_7/attention_output/add:z:0*
T0*+
_output_shapes
:���������>�
=model_6/layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
+model_6/layer_normalization_14/moments/meanMeanmodel_6/add_14/add:z:0Fmodel_6/layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(�
3model_6/layer_normalization_14/moments/StopGradientStopGradient4model_6/layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:���������>�
8model_6/layer_normalization_14/moments/SquaredDifferenceSquaredDifferencemodel_6/add_14/add:z:0<model_6/layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������>�
Amodel_6/layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
/model_6/layer_normalization_14/moments/varianceMean<model_6/layer_normalization_14/moments/SquaredDifference:z:0Jmodel_6/layer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(s
.model_6/layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
,model_6/layer_normalization_14/batchnorm/addAddV28model_6/layer_normalization_14/moments/variance:output:07model_6/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_14/batchnorm/RsqrtRsqrt0model_6/layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:���������>�
;model_6/layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_6_layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
,model_6/layer_normalization_14/batchnorm/mulMul2model_6/layer_normalization_14/batchnorm/Rsqrt:y:0Cmodel_6/layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_14/batchnorm/mul_1Mulmodel_6/add_14/add:z:00model_6/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_14/batchnorm/mul_2Mul4model_6/layer_normalization_14/moments/mean:output:00model_6/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>�
7model_6/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp@model_6_layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
,model_6/layer_normalization_14/batchnorm/subSub?model_6/layer_normalization_14/batchnorm/ReadVariableOp:value:02model_6/layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_14/batchnorm/add_1AddV22model_6/layer_normalization_14/batchnorm/mul_1:z:00model_6/layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������>�
)model_6/dense_20/Tensordot/ReadVariableOpReadVariableOp2model_6_dense_20_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0i
model_6/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
model_6/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
 model_6/dense_20/Tensordot/ShapeShape2model_6/layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
::��j
(model_6/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#model_6/dense_20/Tensordot/GatherV2GatherV2)model_6/dense_20/Tensordot/Shape:output:0(model_6/dense_20/Tensordot/free:output:01model_6/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*model_6/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%model_6/dense_20/Tensordot/GatherV2_1GatherV2)model_6/dense_20/Tensordot/Shape:output:0(model_6/dense_20/Tensordot/axes:output:03model_6/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 model_6/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model_6/dense_20/Tensordot/ProdProd,model_6/dense_20/Tensordot/GatherV2:output:0)model_6/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"model_6/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!model_6/dense_20/Tensordot/Prod_1Prod.model_6/dense_20/Tensordot/GatherV2_1:output:0+model_6/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&model_6/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!model_6/dense_20/Tensordot/concatConcatV2(model_6/dense_20/Tensordot/free:output:0(model_6/dense_20/Tensordot/axes:output:0/model_6/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 model_6/dense_20/Tensordot/stackPack(model_6/dense_20/Tensordot/Prod:output:0*model_6/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$model_6/dense_20/Tensordot/transpose	Transpose2model_6/layer_normalization_14/batchnorm/add_1:z:0*model_6/dense_20/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������>�
"model_6/dense_20/Tensordot/ReshapeReshape(model_6/dense_20/Tensordot/transpose:y:0)model_6/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!model_6/dense_20/Tensordot/MatMulMatMul+model_6/dense_20/Tensordot/Reshape:output:01model_6/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
"model_6/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(model_6/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#model_6/dense_20/Tensordot/concat_1ConcatV2,model_6/dense_20/Tensordot/GatherV2:output:0+model_6/dense_20/Tensordot/Const_2:output:01model_6/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model_6/dense_20/TensordotReshape+model_6/dense_20/Tensordot/MatMul:product:0,model_6/dense_20/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������>�
6model_6/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp?model_6_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0r
-model_6/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model_6/batch_normalization_1/batchnorm/addAddV2>model_6/batch_normalization_1/batchnorm/ReadVariableOp:value:06model_6/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
-model_6/batch_normalization_1/batchnorm/RsqrtRsqrt/model_6/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:�
:model_6/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_6_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
+model_6/batch_normalization_1/batchnorm/mulMul1model_6/batch_normalization_1/batchnorm/Rsqrt:y:0Bmodel_6/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
-model_6/batch_normalization_1/batchnorm/mul_1Mul#model_6/dense_20/Tensordot:output:0/model_6/batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>�
8model_6/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_6_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
-model_6/batch_normalization_1/batchnorm/mul_2Mul@model_6/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0/model_6/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
8model_6/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_6_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
+model_6/batch_normalization_1/batchnorm/subSub@model_6/batch_normalization_1/batchnorm/ReadVariableOp_2:value:01model_6/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
-model_6/batch_normalization_1/batchnorm/add_1AddV21model_6/batch_normalization_1/batchnorm/mul_1:z:0/model_6/batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������>�
model_6/activation_1/ReluRelu1model_6/batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������>�
model_6/dropout_1/IdentityIdentity'model_6/activation_1/Relu:activations:0*
T0*+
_output_shapes
:���������>�
)model_6/dense_21/Tensordot/ReadVariableOpReadVariableOp2model_6_dense_21_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0i
model_6/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
model_6/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
 model_6/dense_21/Tensordot/ShapeShape#model_6/dropout_1/Identity:output:0*
T0*
_output_shapes
::��j
(model_6/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#model_6/dense_21/Tensordot/GatherV2GatherV2)model_6/dense_21/Tensordot/Shape:output:0(model_6/dense_21/Tensordot/free:output:01model_6/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*model_6/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%model_6/dense_21/Tensordot/GatherV2_1GatherV2)model_6/dense_21/Tensordot/Shape:output:0(model_6/dense_21/Tensordot/axes:output:03model_6/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 model_6/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model_6/dense_21/Tensordot/ProdProd,model_6/dense_21/Tensordot/GatherV2:output:0)model_6/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"model_6/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!model_6/dense_21/Tensordot/Prod_1Prod.model_6/dense_21/Tensordot/GatherV2_1:output:0+model_6/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&model_6/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!model_6/dense_21/Tensordot/concatConcatV2(model_6/dense_21/Tensordot/free:output:0(model_6/dense_21/Tensordot/axes:output:0/model_6/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 model_6/dense_21/Tensordot/stackPack(model_6/dense_21/Tensordot/Prod:output:0*model_6/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$model_6/dense_21/Tensordot/transpose	Transpose#model_6/dropout_1/Identity:output:0*model_6/dense_21/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������>�
"model_6/dense_21/Tensordot/ReshapeReshape(model_6/dense_21/Tensordot/transpose:y:0)model_6/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!model_6/dense_21/Tensordot/MatMulMatMul+model_6/dense_21/Tensordot/Reshape:output:01model_6/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
"model_6/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(model_6/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#model_6/dense_21/Tensordot/concat_1ConcatV2,model_6/dense_21/Tensordot/GatherV2:output:0+model_6/dense_21/Tensordot/Const_2:output:01model_6/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model_6/dense_21/TensordotReshape+model_6/dense_21/Tensordot/MatMul:product:0,model_6/dense_21/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������>�
'model_6/dense_21/BiasAdd/ReadVariableOpReadVariableOp0model_6_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_6/dense_21/BiasAddBiasAdd#model_6/dense_21/Tensordot:output:0/model_6/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>�
model_6/add_15/addAddV22model_6/layer_normalization_14/batchnorm/add_1:z:0!model_6/dense_21/BiasAdd:output:0*
T0*+
_output_shapes
:���������>�
=model_6/layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
+model_6/layer_normalization_15/moments/meanMeanmodel_6/add_15/add:z:0Fmodel_6/layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(�
3model_6/layer_normalization_15/moments/StopGradientStopGradient4model_6/layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:���������>�
8model_6/layer_normalization_15/moments/SquaredDifferenceSquaredDifferencemodel_6/add_15/add:z:0<model_6/layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������>�
Amodel_6/layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
/model_6/layer_normalization_15/moments/varianceMean<model_6/layer_normalization_15/moments/SquaredDifference:z:0Jmodel_6/layer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(s
.model_6/layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
,model_6/layer_normalization_15/batchnorm/addAddV28model_6/layer_normalization_15/moments/variance:output:07model_6/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_15/batchnorm/RsqrtRsqrt0model_6/layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:���������>�
;model_6/layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_6_layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
,model_6/layer_normalization_15/batchnorm/mulMul2model_6/layer_normalization_15/batchnorm/Rsqrt:y:0Cmodel_6/layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_15/batchnorm/mul_1Mulmodel_6/add_15/add:z:00model_6/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_15/batchnorm/mul_2Mul4model_6/layer_normalization_15/moments/mean:output:00model_6/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>�
7model_6/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp@model_6_layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
,model_6/layer_normalization_15/batchnorm/subSub?model_6/layer_normalization_15/batchnorm/ReadVariableOp:value:02model_6/layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������>�
.model_6/layer_normalization_15/batchnorm/add_1AddV22model_6/layer_normalization_15/batchnorm/mul_1:z:00model_6/layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������>{
9model_6/global_average_pooling1d_6/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'model_6/global_average_pooling1d_6/MeanMean2model_6/layer_normalization_15/batchnorm/add_1:z:0Bmodel_6/global_average_pooling1d_6/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:����������
&model_6/dense_22/MatMul/ReadVariableOpReadVariableOp/model_6_dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
model_6/dense_22/MatMulMatMul0model_6/global_average_pooling1d_6/Mean:output:0.model_6/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_6/dense_22/BiasAdd/ReadVariableOpReadVariableOp0model_6_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_6/dense_22/BiasAddBiasAdd!model_6/dense_22/MatMul:product:0/model_6/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
model_6/dense_22/SigmoidSigmoid!model_6/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������k
IdentityIdentitymodel_6/dense_22/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp5^model_6/batch_normalization/batchnorm/ReadVariableOp7^model_6/batch_normalization/batchnorm/ReadVariableOp_17^model_6/batch_normalization/batchnorm/ReadVariableOp_29^model_6/batch_normalization/batchnorm/mul/ReadVariableOp7^model_6/batch_normalization_1/batchnorm/ReadVariableOp9^model_6/batch_normalization_1/batchnorm/ReadVariableOp_19^model_6/batch_normalization_1/batchnorm/ReadVariableOp_2;^model_6/batch_normalization_1/batchnorm/mul/ReadVariableOp*^model_6/dense_18/Tensordot/ReadVariableOp(^model_6/dense_19/BiasAdd/ReadVariableOp*^model_6/dense_19/Tensordot/ReadVariableOp*^model_6/dense_20/Tensordot/ReadVariableOp(^model_6/dense_21/BiasAdd/ReadVariableOp*^model_6/dense_21/Tensordot/ReadVariableOp(^model_6/dense_22/BiasAdd/ReadVariableOp'^model_6/dense_22/MatMul/ReadVariableOp8^model_6/layer_normalization_12/batchnorm/ReadVariableOp<^model_6/layer_normalization_12/batchnorm/mul/ReadVariableOp8^model_6/layer_normalization_13/batchnorm/ReadVariableOp<^model_6/layer_normalization_13/batchnorm/mul/ReadVariableOp8^model_6/layer_normalization_14/batchnorm/ReadVariableOp<^model_6/layer_normalization_14/batchnorm/mul/ReadVariableOp8^model_6/layer_normalization_15/batchnorm/ReadVariableOp<^model_6/layer_normalization_15/batchnorm/mul/ReadVariableOpC^model_6/multi_head_attention_6/attention_output/add/ReadVariableOpM^model_6/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp6^model_6/multi_head_attention_6/key/add/ReadVariableOp@^model_6/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp8^model_6/multi_head_attention_6/query/add/ReadVariableOpB^model_6/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp8^model_6/multi_head_attention_6/value/add/ReadVariableOpB^model_6/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpC^model_6/multi_head_attention_7/attention_output/add/ReadVariableOpM^model_6/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp6^model_6/multi_head_attention_7/key/add/ReadVariableOp@^model_6/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp8^model_6/multi_head_attention_7/query/add/ReadVariableOpB^model_6/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp8^model_6/multi_head_attention_7/value/add/ReadVariableOpB^model_6/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesw
u:���������>:>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6model_6/batch_normalization/batchnorm/ReadVariableOp_16model_6/batch_normalization/batchnorm/ReadVariableOp_12p
6model_6/batch_normalization/batchnorm/ReadVariableOp_26model_6/batch_normalization/batchnorm/ReadVariableOp_22l
4model_6/batch_normalization/batchnorm/ReadVariableOp4model_6/batch_normalization/batchnorm/ReadVariableOp2t
8model_6/batch_normalization/batchnorm/mul/ReadVariableOp8model_6/batch_normalization/batchnorm/mul/ReadVariableOp2t
8model_6/batch_normalization_1/batchnorm/ReadVariableOp_18model_6/batch_normalization_1/batchnorm/ReadVariableOp_12t
8model_6/batch_normalization_1/batchnorm/ReadVariableOp_28model_6/batch_normalization_1/batchnorm/ReadVariableOp_22p
6model_6/batch_normalization_1/batchnorm/ReadVariableOp6model_6/batch_normalization_1/batchnorm/ReadVariableOp2x
:model_6/batch_normalization_1/batchnorm/mul/ReadVariableOp:model_6/batch_normalization_1/batchnorm/mul/ReadVariableOp2V
)model_6/dense_18/Tensordot/ReadVariableOp)model_6/dense_18/Tensordot/ReadVariableOp2R
'model_6/dense_19/BiasAdd/ReadVariableOp'model_6/dense_19/BiasAdd/ReadVariableOp2V
)model_6/dense_19/Tensordot/ReadVariableOp)model_6/dense_19/Tensordot/ReadVariableOp2V
)model_6/dense_20/Tensordot/ReadVariableOp)model_6/dense_20/Tensordot/ReadVariableOp2R
'model_6/dense_21/BiasAdd/ReadVariableOp'model_6/dense_21/BiasAdd/ReadVariableOp2V
)model_6/dense_21/Tensordot/ReadVariableOp)model_6/dense_21/Tensordot/ReadVariableOp2R
'model_6/dense_22/BiasAdd/ReadVariableOp'model_6/dense_22/BiasAdd/ReadVariableOp2P
&model_6/dense_22/MatMul/ReadVariableOp&model_6/dense_22/MatMul/ReadVariableOp2r
7model_6/layer_normalization_12/batchnorm/ReadVariableOp7model_6/layer_normalization_12/batchnorm/ReadVariableOp2z
;model_6/layer_normalization_12/batchnorm/mul/ReadVariableOp;model_6/layer_normalization_12/batchnorm/mul/ReadVariableOp2r
7model_6/layer_normalization_13/batchnorm/ReadVariableOp7model_6/layer_normalization_13/batchnorm/ReadVariableOp2z
;model_6/layer_normalization_13/batchnorm/mul/ReadVariableOp;model_6/layer_normalization_13/batchnorm/mul/ReadVariableOp2r
7model_6/layer_normalization_14/batchnorm/ReadVariableOp7model_6/layer_normalization_14/batchnorm/ReadVariableOp2z
;model_6/layer_normalization_14/batchnorm/mul/ReadVariableOp;model_6/layer_normalization_14/batchnorm/mul/ReadVariableOp2r
7model_6/layer_normalization_15/batchnorm/ReadVariableOp7model_6/layer_normalization_15/batchnorm/ReadVariableOp2z
;model_6/layer_normalization_15/batchnorm/mul/ReadVariableOp;model_6/layer_normalization_15/batchnorm/mul/ReadVariableOp2�
Bmodel_6/multi_head_attention_6/attention_output/add/ReadVariableOpBmodel_6/multi_head_attention_6/attention_output/add/ReadVariableOp2�
Lmodel_6/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpLmodel_6/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp2n
5model_6/multi_head_attention_6/key/add/ReadVariableOp5model_6/multi_head_attention_6/key/add/ReadVariableOp2�
?model_6/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp?model_6/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp2r
7model_6/multi_head_attention_6/query/add/ReadVariableOp7model_6/multi_head_attention_6/query/add/ReadVariableOp2�
Amodel_6/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpAmodel_6/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp2r
7model_6/multi_head_attention_6/value/add/ReadVariableOp7model_6/multi_head_attention_6/value/add/ReadVariableOp2�
Amodel_6/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpAmodel_6/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp2�
Bmodel_6/multi_head_attention_7/attention_output/add/ReadVariableOpBmodel_6/multi_head_attention_7/attention_output/add/ReadVariableOp2�
Lmodel_6/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpLmodel_6/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2n
5model_6/multi_head_attention_7/key/add/ReadVariableOp5model_6/multi_head_attention_7/key/add/ReadVariableOp2�
?model_6/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp?model_6/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2r
7model_6/multi_head_attention_7/query/add/ReadVariableOp7model_6/multi_head_attention_7/query/add/ReadVariableOp2�
Amodel_6/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpAmodel_6/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2r
7model_6/multi_head_attention_7/value/add/ReadVariableOp7model_6/multi_head_attention_7/value/add/ReadVariableOp2�
Amodel_6/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpAmodel_6/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp:()$
"
_user_specified_name
resource:(($
"
_user_specified_name
resource:('$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:JF
"
_output_shapes
:>
 
_user_specified_name514653:T P
+
_output_shapes
:���������>
!
_user_specified_name	input_7
�
�
R__inference_layer_normalization_12_layer_call_and_return_conditional_losses_516481

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������>�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������>l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������>a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������>~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������>f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������>\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�!
�	
(__inference_model_6_layer_call_fn_515970
input_7
unknown
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22: 

unknown_23:

unknown_24: 

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_515796o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesw
u:���������>:>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&)"
 
_user_specified_name515966:&("
 
_user_specified_name515964:&'"
 
_user_specified_name515962:&&"
 
_user_specified_name515960:&%"
 
_user_specified_name515958:&$"
 
_user_specified_name515956:&#"
 
_user_specified_name515954:&""
 
_user_specified_name515952:&!"
 
_user_specified_name515950:& "
 
_user_specified_name515948:&"
 
_user_specified_name515946:&"
 
_user_specified_name515944:&"
 
_user_specified_name515942:&"
 
_user_specified_name515940:&"
 
_user_specified_name515938:&"
 
_user_specified_name515936:&"
 
_user_specified_name515934:&"
 
_user_specified_name515932:&"
 
_user_specified_name515930:&"
 
_user_specified_name515928:&"
 
_user_specified_name515926:&"
 
_user_specified_name515924:&"
 
_user_specified_name515922:&"
 
_user_specified_name515920:&"
 
_user_specified_name515918:&"
 
_user_specified_name515916:&"
 
_user_specified_name515914:&"
 
_user_specified_name515912:&"
 
_user_specified_name515910:&"
 
_user_specified_name515908:&"
 
_user_specified_name515906:&
"
 
_user_specified_name515904:&	"
 
_user_specified_name515902:&"
 
_user_specified_name515900:&"
 
_user_specified_name515898:&"
 
_user_specified_name515896:&"
 
_user_specified_name515894:&"
 
_user_specified_name515892:&"
 
_user_specified_name515890:&"
 
_user_specified_name515888:JF
"
_output_shapes
:>
 
_user_specified_name515886:T P
+
_output_shapes
:���������>
!
_user_specified_name	input_7
�
b
F__inference_activation_layer_call_and_return_conditional_losses_516605

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:���������>^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
_
6__inference_positional_encoding_6_layer_call_fn_516293

inputs
unknown
identity�
PartitionedCallPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_positional_encoding_6_layer_call_and_return_conditional_losses_515145d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������>:>:JF
"
_output_shapes
:>
 
_user_specified_name516289:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
R__inference_layer_normalization_14_layer_call_and_return_conditional_losses_515456

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������>�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������>l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������>a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������>~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������>f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������>\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
S
'__inference_add_12_layer_call_fn_516444
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_add_12_layer_call_and_return_conditional_losses_515213d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������>:���������>:UQ
+
_output_shapes
:���������>
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������>
"
_user_specified_name
inputs_0
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_515776

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������>_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������>"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
R__inference_layer_normalization_13_layer_call_and_return_conditional_losses_515363

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������>�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������>l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������>a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������>~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������>f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������>\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
D__inference_dense_18_layer_call_and_return_conditional_losses_515268

inputs3
!tensordot_readvariableop_resource:
identity��Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������>�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������>e
IdentityIdentityTensordot:output:0^NoOp*
T0*+
_output_shapes
:���������>=
NoOpNoOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������>: 24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
l
B__inference_add_12_layer_call_and_return_conditional_losses_515213

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������>S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������>:���������>:SO
+
_output_shapes
:���������>
 
_user_specified_nameinputs:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
d
H__inference_activation_1_layer_call_and_return_conditional_losses_515505

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:���������>^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
n
B__inference_add_13_layer_call_and_return_conditional_losses_516683
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������>S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������>:���������>:UQ
+
_output_shapes
:���������>
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������>
"
_user_specified_name
inputs_0
�
�
7__inference_multi_head_attention_7_layer_call_fn_516736	
query	
value
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_515410s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������>:���������>: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&	"
 
_user_specified_name516732:&"
 
_user_specified_name516730:&"
 
_user_specified_name516728:&"
 
_user_specified_name516726:&"
 
_user_specified_name516724:&"
 
_user_specified_name516722:&"
 
_user_specified_name516720:&"
 
_user_specified_name516718:RN
+
_output_shapes
:���������>

_user_specified_namevalue:R N
+
_output_shapes
:���������>

_user_specified_namequery
�,
�
R__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_516835	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:͓>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������>�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������>>*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������>>q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������>>�
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������>*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������>*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:���������>�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������>:���������>: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������>

_user_specified_namevalue:R N
+
_output_shapes
:���������>

_user_specified_namequery
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_515082

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_515298

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������>Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������>*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������>T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:���������>e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
R__inference_layer_normalization_12_layer_call_and_return_conditional_losses_515236

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������>�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������>l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������>a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������>~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������>f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������>\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�4
�
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_515190	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:͓>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������>�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������>>*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������>>Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������>>l
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
::���
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������>>*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������>>\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������>>�
einsum_1/EinsumEinsum!dropout/dropout/SelectV2:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������>*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������>*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:���������>�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������>:���������>: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������>

_user_specified_namevalue:R N
+
_output_shapes
:���������>

_user_specified_namequery
�
b
F__inference_activation_layer_call_and_return_conditional_losses_515285

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:���������>^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
n
B__inference_add_15_layer_call_and_return_conditional_losses_517080
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������>S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������>:���������>:UQ
+
_output_shapes
:���������>
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������>
"
_user_specified_name
inputs_0
�
�
R__inference_layer_normalization_13_layer_call_and_return_conditional_losses_516714

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������>�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������>l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������>a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������>~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������>f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������>\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
S
'__inference_add_15_layer_call_fn_517074
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_add_15_layer_call_and_return_conditional_losses_515560d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������>:���������>:UQ
+
_output_shapes
:���������>
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������>
"
_user_specified_name
inputs_0
�

�
D__inference_dense_22_layer_call_and_return_conditional_losses_515600

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_517029

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������>_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������>"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
r
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_515114

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
S
'__inference_add_13_layer_call_fn_516677
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_add_13_layer_call_and_return_conditional_losses_515340d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������>:���������>:UQ
+
_output_shapes
:���������>
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������>
"
_user_specified_name
inputs_0
�

�
D__inference_dense_22_layer_call_and_return_conditional_losses_517142

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�U
"__inference__traced_restore_518236
file_prefix;
-assignvariableop_layer_normalization_12_gamma:<
.assignvariableop_1_layer_normalization_12_beta:4
"assignvariableop_2_dense_18_kernel::
,assignvariableop_3_batch_normalization_gamma:9
+assignvariableop_4_batch_normalization_beta:@
2assignvariableop_5_batch_normalization_moving_mean:D
6assignvariableop_6_batch_normalization_moving_variance:4
"assignvariableop_7_dense_19_kernel:.
 assignvariableop_8_dense_19_bias:=
/assignvariableop_9_layer_normalization_13_gamma:=
/assignvariableop_10_layer_normalization_13_beta:>
0assignvariableop_11_layer_normalization_14_gamma:=
/assignvariableop_12_layer_normalization_14_beta:5
#assignvariableop_13_dense_20_kernel:=
/assignvariableop_14_batch_normalization_1_gamma:<
.assignvariableop_15_batch_normalization_1_beta:C
5assignvariableop_16_batch_normalization_1_moving_mean:G
9assignvariableop_17_batch_normalization_1_moving_variance:5
#assignvariableop_18_dense_21_kernel:/
!assignvariableop_19_dense_21_bias:>
0assignvariableop_20_layer_normalization_15_gamma:=
/assignvariableop_21_layer_normalization_15_beta:5
#assignvariableop_22_dense_22_kernel:/
!assignvariableop_23_dense_22_bias:M
7assignvariableop_24_multi_head_attention_6_query_kernel:G
5assignvariableop_25_multi_head_attention_6_query_bias:K
5assignvariableop_26_multi_head_attention_6_key_kernel:E
3assignvariableop_27_multi_head_attention_6_key_bias:M
7assignvariableop_28_multi_head_attention_6_value_kernel:G
5assignvariableop_29_multi_head_attention_6_value_bias:X
Bassignvariableop_30_multi_head_attention_6_attention_output_kernel:N
@assignvariableop_31_multi_head_attention_6_attention_output_bias:M
7assignvariableop_32_multi_head_attention_7_query_kernel:G
5assignvariableop_33_multi_head_attention_7_query_bias:K
5assignvariableop_34_multi_head_attention_7_key_kernel:E
3assignvariableop_35_multi_head_attention_7_key_bias:M
7assignvariableop_36_multi_head_attention_7_value_kernel:G
5assignvariableop_37_multi_head_attention_7_value_bias:X
Bassignvariableop_38_multi_head_attention_7_attention_output_kernel:N
@assignvariableop_39_multi_head_attention_7_attention_output_bias:'
assignvariableop_40_iteration:	 +
!assignvariableop_41_learning_rate: T
>assignvariableop_42_adam_m_multi_head_attention_6_query_kernel:T
>assignvariableop_43_adam_v_multi_head_attention_6_query_kernel:N
<assignvariableop_44_adam_m_multi_head_attention_6_query_bias:N
<assignvariableop_45_adam_v_multi_head_attention_6_query_bias:R
<assignvariableop_46_adam_m_multi_head_attention_6_key_kernel:R
<assignvariableop_47_adam_v_multi_head_attention_6_key_kernel:L
:assignvariableop_48_adam_m_multi_head_attention_6_key_bias:L
:assignvariableop_49_adam_v_multi_head_attention_6_key_bias:T
>assignvariableop_50_adam_m_multi_head_attention_6_value_kernel:T
>assignvariableop_51_adam_v_multi_head_attention_6_value_kernel:N
<assignvariableop_52_adam_m_multi_head_attention_6_value_bias:N
<assignvariableop_53_adam_v_multi_head_attention_6_value_bias:_
Iassignvariableop_54_adam_m_multi_head_attention_6_attention_output_kernel:_
Iassignvariableop_55_adam_v_multi_head_attention_6_attention_output_kernel:U
Gassignvariableop_56_adam_m_multi_head_attention_6_attention_output_bias:U
Gassignvariableop_57_adam_v_multi_head_attention_6_attention_output_bias:E
7assignvariableop_58_adam_m_layer_normalization_12_gamma:E
7assignvariableop_59_adam_v_layer_normalization_12_gamma:D
6assignvariableop_60_adam_m_layer_normalization_12_beta:D
6assignvariableop_61_adam_v_layer_normalization_12_beta:<
*assignvariableop_62_adam_m_dense_18_kernel:<
*assignvariableop_63_adam_v_dense_18_kernel:B
4assignvariableop_64_adam_m_batch_normalization_gamma:B
4assignvariableop_65_adam_v_batch_normalization_gamma:A
3assignvariableop_66_adam_m_batch_normalization_beta:A
3assignvariableop_67_adam_v_batch_normalization_beta:<
*assignvariableop_68_adam_m_dense_19_kernel:<
*assignvariableop_69_adam_v_dense_19_kernel:6
(assignvariableop_70_adam_m_dense_19_bias:6
(assignvariableop_71_adam_v_dense_19_bias:E
7assignvariableop_72_adam_m_layer_normalization_13_gamma:E
7assignvariableop_73_adam_v_layer_normalization_13_gamma:D
6assignvariableop_74_adam_m_layer_normalization_13_beta:D
6assignvariableop_75_adam_v_layer_normalization_13_beta:T
>assignvariableop_76_adam_m_multi_head_attention_7_query_kernel:T
>assignvariableop_77_adam_v_multi_head_attention_7_query_kernel:N
<assignvariableop_78_adam_m_multi_head_attention_7_query_bias:N
<assignvariableop_79_adam_v_multi_head_attention_7_query_bias:R
<assignvariableop_80_adam_m_multi_head_attention_7_key_kernel:R
<assignvariableop_81_adam_v_multi_head_attention_7_key_kernel:L
:assignvariableop_82_adam_m_multi_head_attention_7_key_bias:L
:assignvariableop_83_adam_v_multi_head_attention_7_key_bias:T
>assignvariableop_84_adam_m_multi_head_attention_7_value_kernel:T
>assignvariableop_85_adam_v_multi_head_attention_7_value_kernel:N
<assignvariableop_86_adam_m_multi_head_attention_7_value_bias:N
<assignvariableop_87_adam_v_multi_head_attention_7_value_bias:_
Iassignvariableop_88_adam_m_multi_head_attention_7_attention_output_kernel:_
Iassignvariableop_89_adam_v_multi_head_attention_7_attention_output_kernel:U
Gassignvariableop_90_adam_m_multi_head_attention_7_attention_output_bias:U
Gassignvariableop_91_adam_v_multi_head_attention_7_attention_output_bias:E
7assignvariableop_92_adam_m_layer_normalization_14_gamma:E
7assignvariableop_93_adam_v_layer_normalization_14_gamma:D
6assignvariableop_94_adam_m_layer_normalization_14_beta:D
6assignvariableop_95_adam_v_layer_normalization_14_beta:<
*assignvariableop_96_adam_m_dense_20_kernel:<
*assignvariableop_97_adam_v_dense_20_kernel:D
6assignvariableop_98_adam_m_batch_normalization_1_gamma:D
6assignvariableop_99_adam_v_batch_normalization_1_gamma:D
6assignvariableop_100_adam_m_batch_normalization_1_beta:D
6assignvariableop_101_adam_v_batch_normalization_1_beta:=
+assignvariableop_102_adam_m_dense_21_kernel:=
+assignvariableop_103_adam_v_dense_21_kernel:7
)assignvariableop_104_adam_m_dense_21_bias:7
)assignvariableop_105_adam_v_dense_21_bias:F
8assignvariableop_106_adam_m_layer_normalization_15_gamma:F
8assignvariableop_107_adam_v_layer_normalization_15_gamma:E
7assignvariableop_108_adam_m_layer_normalization_15_beta:E
7assignvariableop_109_adam_v_layer_normalization_15_beta:=
+assignvariableop_110_adam_m_dense_22_kernel:=
+assignvariableop_111_adam_v_dense_22_kernel:7
)assignvariableop_112_adam_m_dense_22_bias:7
)assignvariableop_113_adam_v_dense_22_bias:&
assignvariableop_114_total_1: &
assignvariableop_115_count_1: $
assignvariableop_116_total: $
assignvariableop_117_count: 
identity_119��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�0
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:w*
dtype0*�/
value�/B�/wB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:w*
dtype0*�
value�B�wB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes{
y2w	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp-assignvariableop_layer_normalization_12_gammaIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp.assignvariableop_1_layer_normalization_12_betaIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_18_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp,assignvariableop_3_batch_normalization_gammaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp+assignvariableop_4_batch_normalization_betaIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp2assignvariableop_5_batch_normalization_moving_meanIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp6assignvariableop_6_batch_normalization_moving_varianceIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_19_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_19_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp/assignvariableop_9_layer_normalization_13_gammaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_layer_normalization_13_betaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp0assignvariableop_11_layer_normalization_14_gammaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp/assignvariableop_12_layer_normalization_14_betaIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_20_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_1_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_1_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_1_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_1_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_21_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_21_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp0assignvariableop_20_layer_normalization_15_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp/assignvariableop_21_layer_normalization_15_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_22_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_dense_22_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp7assignvariableop_24_multi_head_attention_6_query_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp5assignvariableop_25_multi_head_attention_6_query_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp5assignvariableop_26_multi_head_attention_6_key_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp3assignvariableop_27_multi_head_attention_6_key_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp7assignvariableop_28_multi_head_attention_6_value_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp5assignvariableop_29_multi_head_attention_6_value_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpBassignvariableop_30_multi_head_attention_6_attention_output_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp@assignvariableop_31_multi_head_attention_6_attention_output_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp7assignvariableop_32_multi_head_attention_7_query_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp5assignvariableop_33_multi_head_attention_7_query_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp5assignvariableop_34_multi_head_attention_7_key_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp3assignvariableop_35_multi_head_attention_7_key_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp7assignvariableop_36_multi_head_attention_7_value_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp5assignvariableop_37_multi_head_attention_7_value_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpBassignvariableop_38_multi_head_attention_7_attention_output_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp@assignvariableop_39_multi_head_attention_7_attention_output_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpassignvariableop_40_iterationIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp!assignvariableop_41_learning_rateIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp>assignvariableop_42_adam_m_multi_head_attention_6_query_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp>assignvariableop_43_adam_v_multi_head_attention_6_query_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp<assignvariableop_44_adam_m_multi_head_attention_6_query_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp<assignvariableop_45_adam_v_multi_head_attention_6_query_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp<assignvariableop_46_adam_m_multi_head_attention_6_key_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp<assignvariableop_47_adam_v_multi_head_attention_6_key_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp:assignvariableop_48_adam_m_multi_head_attention_6_key_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp:assignvariableop_49_adam_v_multi_head_attention_6_key_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp>assignvariableop_50_adam_m_multi_head_attention_6_value_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp>assignvariableop_51_adam_v_multi_head_attention_6_value_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp<assignvariableop_52_adam_m_multi_head_attention_6_value_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp<assignvariableop_53_adam_v_multi_head_attention_6_value_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpIassignvariableop_54_adam_m_multi_head_attention_6_attention_output_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpIassignvariableop_55_adam_v_multi_head_attention_6_attention_output_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpGassignvariableop_56_adam_m_multi_head_attention_6_attention_output_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpGassignvariableop_57_adam_v_multi_head_attention_6_attention_output_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp7assignvariableop_58_adam_m_layer_normalization_12_gammaIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_v_layer_normalization_12_gammaIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_m_layer_normalization_12_betaIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_v_layer_normalization_12_betaIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_m_dense_18_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_v_dense_18_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp4assignvariableop_64_adam_m_batch_normalization_gammaIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp4assignvariableop_65_adam_v_batch_normalization_gammaIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp3assignvariableop_66_adam_m_batch_normalization_betaIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp3assignvariableop_67_adam_v_batch_normalization_betaIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_m_dense_19_kernelIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_v_dense_19_kernelIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_m_dense_19_biasIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_v_dense_19_biasIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp7assignvariableop_72_adam_m_layer_normalization_13_gammaIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp7assignvariableop_73_adam_v_layer_normalization_13_gammaIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp6assignvariableop_74_adam_m_layer_normalization_13_betaIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp6assignvariableop_75_adam_v_layer_normalization_13_betaIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp>assignvariableop_76_adam_m_multi_head_attention_7_query_kernelIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp>assignvariableop_77_adam_v_multi_head_attention_7_query_kernelIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp<assignvariableop_78_adam_m_multi_head_attention_7_query_biasIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp<assignvariableop_79_adam_v_multi_head_attention_7_query_biasIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp<assignvariableop_80_adam_m_multi_head_attention_7_key_kernelIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp<assignvariableop_81_adam_v_multi_head_attention_7_key_kernelIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp:assignvariableop_82_adam_m_multi_head_attention_7_key_biasIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp:assignvariableop_83_adam_v_multi_head_attention_7_key_biasIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp>assignvariableop_84_adam_m_multi_head_attention_7_value_kernelIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp>assignvariableop_85_adam_v_multi_head_attention_7_value_kernelIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp<assignvariableop_86_adam_m_multi_head_attention_7_value_biasIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp<assignvariableop_87_adam_v_multi_head_attention_7_value_biasIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOpIassignvariableop_88_adam_m_multi_head_attention_7_attention_output_kernelIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOpIassignvariableop_89_adam_v_multi_head_attention_7_attention_output_kernelIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOpGassignvariableop_90_adam_m_multi_head_attention_7_attention_output_biasIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOpGassignvariableop_91_adam_v_multi_head_attention_7_attention_output_biasIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp7assignvariableop_92_adam_m_layer_normalization_14_gammaIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp7assignvariableop_93_adam_v_layer_normalization_14_gammaIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp6assignvariableop_94_adam_m_layer_normalization_14_betaIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp6assignvariableop_95_adam_v_layer_normalization_14_betaIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp*assignvariableop_96_adam_m_dense_20_kernelIdentity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp*assignvariableop_97_adam_v_dense_20_kernelIdentity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp6assignvariableop_98_adam_m_batch_normalization_1_gammaIdentity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp6assignvariableop_99_adam_v_batch_normalization_1_gammaIdentity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp6assignvariableop_100_adam_m_batch_normalization_1_betaIdentity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp6assignvariableop_101_adam_v_batch_normalization_1_betaIdentity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp+assignvariableop_102_adam_m_dense_21_kernelIdentity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_v_dense_21_kernelIdentity_103:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_m_dense_21_biasIdentity_104:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp)assignvariableop_105_adam_v_dense_21_biasIdentity_105:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp8assignvariableop_106_adam_m_layer_normalization_15_gammaIdentity_106:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOp8assignvariableop_107_adam_v_layer_normalization_15_gammaIdentity_107:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOp7assignvariableop_108_adam_m_layer_normalization_15_betaIdentity_108:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOp7assignvariableop_109_adam_v_layer_normalization_15_betaIdentity_109:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp+assignvariableop_110_adam_m_dense_22_kernelIdentity_110:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOp+assignvariableop_111_adam_v_dense_22_kernelIdentity_111:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp)assignvariableop_112_adam_m_dense_22_biasIdentity_112:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOp)assignvariableop_113_adam_v_dense_22_biasIdentity_113:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOpassignvariableop_114_total_1Identity_114:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_115AssignVariableOpassignvariableop_115_count_1Identity_115:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_116AssignVariableOpassignvariableop_116_totalIdentity_116:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_117AssignVariableOpassignvariableop_117_countIdentity_117:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_118Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_119IdentityIdentity_118:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
_output_shapes
 "%
identity_119Identity_119:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172*
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
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_992(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%v!

_user_specified_namecount:%u!

_user_specified_nametotal:'t#
!
_user_specified_name	count_1:'s#
!
_user_specified_name	total_1:4r0
.
_user_specified_nameAdam/v/dense_22/bias:4q0
.
_user_specified_nameAdam/m/dense_22/bias:6p2
0
_user_specified_nameAdam/v/dense_22/kernel:6o2
0
_user_specified_nameAdam/m/dense_22/kernel:Bn>
<
_user_specified_name$"Adam/v/layer_normalization_15/beta:Bm>
<
_user_specified_name$"Adam/m/layer_normalization_15/beta:Cl?
=
_user_specified_name%#Adam/v/layer_normalization_15/gamma:Ck?
=
_user_specified_name%#Adam/m/layer_normalization_15/gamma:4j0
.
_user_specified_nameAdam/v/dense_21/bias:4i0
.
_user_specified_nameAdam/m/dense_21/bias:6h2
0
_user_specified_nameAdam/v/dense_21/kernel:6g2
0
_user_specified_nameAdam/m/dense_21/kernel:Af=
;
_user_specified_name#!Adam/v/batch_normalization_1/beta:Ae=
;
_user_specified_name#!Adam/m/batch_normalization_1/beta:Bd>
<
_user_specified_name$"Adam/v/batch_normalization_1/gamma:Bc>
<
_user_specified_name$"Adam/m/batch_normalization_1/gamma:6b2
0
_user_specified_nameAdam/v/dense_20/kernel:6a2
0
_user_specified_nameAdam/m/dense_20/kernel:B`>
<
_user_specified_name$"Adam/v/layer_normalization_14/beta:B_>
<
_user_specified_name$"Adam/m/layer_normalization_14/beta:C^?
=
_user_specified_name%#Adam/v/layer_normalization_14/gamma:C]?
=
_user_specified_name%#Adam/m/layer_normalization_14/gamma:S\O
M
_user_specified_name53Adam/v/multi_head_attention_7/attention_output/bias:S[O
M
_user_specified_name53Adam/m/multi_head_attention_7/attention_output/bias:UZQ
O
_user_specified_name75Adam/v/multi_head_attention_7/attention_output/kernel:UYQ
O
_user_specified_name75Adam/m/multi_head_attention_7/attention_output/kernel:HXD
B
_user_specified_name*(Adam/v/multi_head_attention_7/value/bias:HWD
B
_user_specified_name*(Adam/m/multi_head_attention_7/value/bias:JVF
D
_user_specified_name,*Adam/v/multi_head_attention_7/value/kernel:JUF
D
_user_specified_name,*Adam/m/multi_head_attention_7/value/kernel:FTB
@
_user_specified_name(&Adam/v/multi_head_attention_7/key/bias:FSB
@
_user_specified_name(&Adam/m/multi_head_attention_7/key/bias:HRD
B
_user_specified_name*(Adam/v/multi_head_attention_7/key/kernel:HQD
B
_user_specified_name*(Adam/m/multi_head_attention_7/key/kernel:HPD
B
_user_specified_name*(Adam/v/multi_head_attention_7/query/bias:HOD
B
_user_specified_name*(Adam/m/multi_head_attention_7/query/bias:JNF
D
_user_specified_name,*Adam/v/multi_head_attention_7/query/kernel:JMF
D
_user_specified_name,*Adam/m/multi_head_attention_7/query/kernel:BL>
<
_user_specified_name$"Adam/v/layer_normalization_13/beta:BK>
<
_user_specified_name$"Adam/m/layer_normalization_13/beta:CJ?
=
_user_specified_name%#Adam/v/layer_normalization_13/gamma:CI?
=
_user_specified_name%#Adam/m/layer_normalization_13/gamma:4H0
.
_user_specified_nameAdam/v/dense_19/bias:4G0
.
_user_specified_nameAdam/m/dense_19/bias:6F2
0
_user_specified_nameAdam/v/dense_19/kernel:6E2
0
_user_specified_nameAdam/m/dense_19/kernel:?D;
9
_user_specified_name!Adam/v/batch_normalization/beta:?C;
9
_user_specified_name!Adam/m/batch_normalization/beta:@B<
:
_user_specified_name" Adam/v/batch_normalization/gamma:@A<
:
_user_specified_name" Adam/m/batch_normalization/gamma:6@2
0
_user_specified_nameAdam/v/dense_18/kernel:6?2
0
_user_specified_nameAdam/m/dense_18/kernel:B>>
<
_user_specified_name$"Adam/v/layer_normalization_12/beta:B=>
<
_user_specified_name$"Adam/m/layer_normalization_12/beta:C<?
=
_user_specified_name%#Adam/v/layer_normalization_12/gamma:C;?
=
_user_specified_name%#Adam/m/layer_normalization_12/gamma:S:O
M
_user_specified_name53Adam/v/multi_head_attention_6/attention_output/bias:S9O
M
_user_specified_name53Adam/m/multi_head_attention_6/attention_output/bias:U8Q
O
_user_specified_name75Adam/v/multi_head_attention_6/attention_output/kernel:U7Q
O
_user_specified_name75Adam/m/multi_head_attention_6/attention_output/kernel:H6D
B
_user_specified_name*(Adam/v/multi_head_attention_6/value/bias:H5D
B
_user_specified_name*(Adam/m/multi_head_attention_6/value/bias:J4F
D
_user_specified_name,*Adam/v/multi_head_attention_6/value/kernel:J3F
D
_user_specified_name,*Adam/m/multi_head_attention_6/value/kernel:F2B
@
_user_specified_name(&Adam/v/multi_head_attention_6/key/bias:F1B
@
_user_specified_name(&Adam/m/multi_head_attention_6/key/bias:H0D
B
_user_specified_name*(Adam/v/multi_head_attention_6/key/kernel:H/D
B
_user_specified_name*(Adam/m/multi_head_attention_6/key/kernel:H.D
B
_user_specified_name*(Adam/v/multi_head_attention_6/query/bias:H-D
B
_user_specified_name*(Adam/m/multi_head_attention_6/query/bias:J,F
D
_user_specified_name,*Adam/v/multi_head_attention_6/query/kernel:J+F
D
_user_specified_name,*Adam/m/multi_head_attention_6/query/kernel:-*)
'
_user_specified_namelearning_rate:))%
#
_user_specified_name	iteration:L(H
F
_user_specified_name.,multi_head_attention_7/attention_output/bias:N'J
H
_user_specified_name0.multi_head_attention_7/attention_output/kernel:A&=
;
_user_specified_name#!multi_head_attention_7/value/bias:C%?
=
_user_specified_name%#multi_head_attention_7/value/kernel:?$;
9
_user_specified_name!multi_head_attention_7/key/bias:A#=
;
_user_specified_name#!multi_head_attention_7/key/kernel:A"=
;
_user_specified_name#!multi_head_attention_7/query/bias:C!?
=
_user_specified_name%#multi_head_attention_7/query/kernel:L H
F
_user_specified_name.,multi_head_attention_6/attention_output/bias:NJ
H
_user_specified_name0.multi_head_attention_6/attention_output/kernel:A=
;
_user_specified_name#!multi_head_attention_6/value/bias:C?
=
_user_specified_name%#multi_head_attention_6/value/kernel:?;
9
_user_specified_name!multi_head_attention_6/key/bias:A=
;
_user_specified_name#!multi_head_attention_6/key/kernel:A=
;
_user_specified_name#!multi_head_attention_6/query/bias:C?
=
_user_specified_name%#multi_head_attention_6/query/kernel:-)
'
_user_specified_namedense_22/bias:/+
)
_user_specified_namedense_22/kernel:;7
5
_user_specified_namelayer_normalization_15/beta:<8
6
_user_specified_namelayer_normalization_15/gamma:-)
'
_user_specified_namedense_21/bias:/+
)
_user_specified_namedense_21/kernel:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean::6
4
_user_specified_namebatch_normalization_1/beta:;7
5
_user_specified_namebatch_normalization_1/gamma:/+
)
_user_specified_namedense_20/kernel:;7
5
_user_specified_namelayer_normalization_14/beta:<8
6
_user_specified_namelayer_normalization_14/gamma:;7
5
_user_specified_namelayer_normalization_13/beta:<
8
6
_user_specified_namelayer_normalization_13/gamma:-	)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:C?
=
_user_specified_name%#batch_normalization/moving_variance:?;
9
_user_specified_name!batch_normalization/moving_mean:84
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:/+
)
_user_specified_namedense_18/kernel:;7
5
_user_specified_namelayer_normalization_12/beta:<8
6
_user_specified_namelayer_normalization_12/gamma:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
)__inference_dense_21_layer_call_fn_517038

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_515549s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name517034:&"
 
_user_specified_name517032:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
}
)__inference_dense_20_layer_call_fn_516885

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_515488s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������>: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name516881:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
C__inference_model_6_layer_call_and_return_conditional_losses_515607
input_7 
positional_encoding_6_5151463
multi_head_attention_6_515191:/
multi_head_attention_6_515193:3
multi_head_attention_6_515195:/
multi_head_attention_6_515197:3
multi_head_attention_6_515199:/
multi_head_attention_6_515201:3
multi_head_attention_6_515203:+
multi_head_attention_6_515205:+
layer_normalization_12_515237:+
layer_normalization_12_515239:!
dense_18_515269:(
batch_normalization_515272:(
batch_normalization_515274:(
batch_normalization_515276:(
batch_normalization_515278:!
dense_19_515330:
dense_19_515332:+
layer_normalization_13_515364:+
layer_normalization_13_515366:3
multi_head_attention_7_515411:/
multi_head_attention_7_515413:3
multi_head_attention_7_515415:/
multi_head_attention_7_515417:3
multi_head_attention_7_515419:/
multi_head_attention_7_515421:3
multi_head_attention_7_515423:+
multi_head_attention_7_515425:+
layer_normalization_14_515457:+
layer_normalization_14_515459:!
dense_20_515489:*
batch_normalization_1_515492:*
batch_normalization_1_515494:*
batch_normalization_1_515496:*
batch_normalization_1_515498:!
dense_21_515550:
dense_21_515552:+
layer_normalization_15_515584:+
layer_normalization_15_515586:!
dense_22_515601:
dense_22_515603:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�.layer_normalization_12/StatefulPartitionedCall�.layer_normalization_13/StatefulPartitionedCall�.layer_normalization_14/StatefulPartitionedCall�.layer_normalization_15/StatefulPartitionedCall�.multi_head_attention_6/StatefulPartitionedCall�.multi_head_attention_7/StatefulPartitionedCall�
%positional_encoding_6/PartitionedCallPartitionedCallinput_7positional_encoding_6_515146*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_positional_encoding_6_layer_call_and_return_conditional_losses_515145�
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall.positional_encoding_6/PartitionedCall:output:0.positional_encoding_6/PartitionedCall:output:0multi_head_attention_6_515191multi_head_attention_6_515193multi_head_attention_6_515195multi_head_attention_6_515197multi_head_attention_6_515199multi_head_attention_6_515201multi_head_attention_6_515203multi_head_attention_6_515205*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_515190�
add_12/PartitionedCallPartitionedCall.positional_encoding_6/PartitionedCall:output:07multi_head_attention_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_add_12_layer_call_and_return_conditional_losses_515213�
.layer_normalization_12/StatefulPartitionedCallStatefulPartitionedCalladd_12/PartitionedCall:output:0layer_normalization_12_515237layer_normalization_12_515239*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_layer_normalization_12_layer_call_and_return_conditional_losses_515236�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:0dense_18_515269*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_515268�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0batch_normalization_515272batch_normalization_515274batch_normalization_515276batch_normalization_515278*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_514982�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_515285�
dropout/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_515298�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_19_515330dense_19_515332*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_515329�
add_13/PartitionedCallPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:0)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_add_13_layer_call_and_return_conditional_losses_515340�
.layer_normalization_13/StatefulPartitionedCallStatefulPartitionedCalladd_13/PartitionedCall:output:0layer_normalization_13_515364layer_normalization_13_515366*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_layer_normalization_13_layer_call_and_return_conditional_losses_515363�
.multi_head_attention_7/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_13/StatefulPartitionedCall:output:07layer_normalization_13/StatefulPartitionedCall:output:0multi_head_attention_7_515411multi_head_attention_7_515413multi_head_attention_7_515415multi_head_attention_7_515417multi_head_attention_7_515419multi_head_attention_7_515421multi_head_attention_7_515423multi_head_attention_7_515425*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_515410�
add_14/PartitionedCallPartitionedCall7layer_normalization_13/StatefulPartitionedCall:output:07multi_head_attention_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_add_14_layer_call_and_return_conditional_losses_515433�
.layer_normalization_14/StatefulPartitionedCallStatefulPartitionedCalladd_14/PartitionedCall:output:0layer_normalization_14_515457layer_normalization_14_515459*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_layer_normalization_14_layer_call_and_return_conditional_losses_515456�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_14/StatefulPartitionedCall:output:0dense_20_515489*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_515488�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_1_515492batch_normalization_1_515494batch_normalization_1_515496batch_normalization_1_515498*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_515062�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_515505�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_515518�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_21_515550dense_21_515552*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_515549�
add_15/PartitionedCallPartitionedCall7layer_normalization_14/StatefulPartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_add_15_layer_call_and_return_conditional_losses_515560�
.layer_normalization_15/StatefulPartitionedCallStatefulPartitionedCalladd_15/PartitionedCall:output:0layer_normalization_15_515584layer_normalization_15_515586*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_515583�
*global_average_pooling1d_6/PartitionedCallPartitionedCall7layer_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *_
fZRX
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_515114�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_6/PartitionedCall:output:0dense_22_515601dense_22_515603*
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
D__inference_dense_22_layer_call_and_return_conditional_losses_515600x
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall/^layer_normalization_12/StatefulPartitionedCall/^layer_normalization_13/StatefulPartitionedCall/^layer_normalization_14/StatefulPartitionedCall/^layer_normalization_15/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall/^multi_head_attention_7/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesw
u:���������>:>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2`
.layer_normalization_12/StatefulPartitionedCall.layer_normalization_12/StatefulPartitionedCall2`
.layer_normalization_13/StatefulPartitionedCall.layer_normalization_13/StatefulPartitionedCall2`
.layer_normalization_14/StatefulPartitionedCall.layer_normalization_14/StatefulPartitionedCall2`
.layer_normalization_15/StatefulPartitionedCall.layer_normalization_15/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2`
.multi_head_attention_7/StatefulPartitionedCall.multi_head_attention_7/StatefulPartitionedCall:&)"
 
_user_specified_name515603:&("
 
_user_specified_name515601:&'"
 
_user_specified_name515586:&&"
 
_user_specified_name515584:&%"
 
_user_specified_name515552:&$"
 
_user_specified_name515550:&#"
 
_user_specified_name515498:&""
 
_user_specified_name515496:&!"
 
_user_specified_name515494:& "
 
_user_specified_name515492:&"
 
_user_specified_name515489:&"
 
_user_specified_name515459:&"
 
_user_specified_name515457:&"
 
_user_specified_name515425:&"
 
_user_specified_name515423:&"
 
_user_specified_name515421:&"
 
_user_specified_name515419:&"
 
_user_specified_name515417:&"
 
_user_specified_name515415:&"
 
_user_specified_name515413:&"
 
_user_specified_name515411:&"
 
_user_specified_name515366:&"
 
_user_specified_name515364:&"
 
_user_specified_name515332:&"
 
_user_specified_name515330:&"
 
_user_specified_name515278:&"
 
_user_specified_name515276:&"
 
_user_specified_name515274:&"
 
_user_specified_name515272:&"
 
_user_specified_name515269:&"
 
_user_specified_name515239:&
"
 
_user_specified_name515237:&	"
 
_user_specified_name515205:&"
 
_user_specified_name515203:&"
 
_user_specified_name515201:&"
 
_user_specified_name515199:&"
 
_user_specified_name515197:&"
 
_user_specified_name515195:&"
 
_user_specified_name515193:&"
 
_user_specified_name515191:JF
"
_output_shapes
:>
 
_user_specified_name515146:T P
+
_output_shapes
:���������>
!
_user_specified_name	input_7
�
�
D__inference_dense_21_layer_call_and_return_conditional_losses_517068

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������>�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������>r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������>V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
)__inference_dense_19_layer_call_fn_516641

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_515329s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name516637:&"
 
_user_specified_name516635:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
)__inference_dense_22_layer_call_fn_517131

inputs
unknown:
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
D__inference_dense_22_layer_call_and_return_conditional_losses_515600o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name517127:&"
 
_user_specified_name517125:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�4
�
R__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_516800	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:͓>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������>�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������>>*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������>>Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������>>l
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
::���
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������>>*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������>>\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������>>�
einsum_1/EinsumEinsum!dropout/dropout/SelectV2:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������>*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������>*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:���������>�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������>:���������>: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������>

_user_specified_namevalue:R N
+
_output_shapes
:���������>

_user_specified_namequery
�!
�	
$__inference_signature_wrapper_516286
input_7
unknown
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18: 

unknown_19:

unknown_20: 

unknown_21:

unknown_22: 

unknown_23:

unknown_24: 

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_514948o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesw
u:���������>:>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&)"
 
_user_specified_name516282:&("
 
_user_specified_name516280:&'"
 
_user_specified_name516278:&&"
 
_user_specified_name516276:&%"
 
_user_specified_name516274:&$"
 
_user_specified_name516272:&#"
 
_user_specified_name516270:&""
 
_user_specified_name516268:&!"
 
_user_specified_name516266:& "
 
_user_specified_name516264:&"
 
_user_specified_name516262:&"
 
_user_specified_name516260:&"
 
_user_specified_name516258:&"
 
_user_specified_name516256:&"
 
_user_specified_name516254:&"
 
_user_specified_name516252:&"
 
_user_specified_name516250:&"
 
_user_specified_name516248:&"
 
_user_specified_name516246:&"
 
_user_specified_name516244:&"
 
_user_specified_name516242:&"
 
_user_specified_name516240:&"
 
_user_specified_name516238:&"
 
_user_specified_name516236:&"
 
_user_specified_name516234:&"
 
_user_specified_name516232:&"
 
_user_specified_name516230:&"
 
_user_specified_name516228:&"
 
_user_specified_name516226:&"
 
_user_specified_name516224:&"
 
_user_specified_name516222:&
"
 
_user_specified_name516220:&	"
 
_user_specified_name516218:&"
 
_user_specified_name516216:&"
 
_user_specified_name516214:&"
 
_user_specified_name516212:&"
 
_user_specified_name516210:&"
 
_user_specified_name516208:&"
 
_user_specified_name516206:&"
 
_user_specified_name516204:JF
"
_output_shapes
:>
 
_user_specified_name516202:T P
+
_output_shapes
:���������>
!
_user_specified_name	input_7
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_516595

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
G
+__inference_activation_layer_call_fn_516600

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_515285d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
W
;__inference_global_average_pooling1d_6_layer_call_fn_517116

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *_
fZRX
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_515114i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_517024

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������>Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������>*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������>T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:���������>e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_1_layer_call_fn_516925

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_515062|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name516921:&"
 
_user_specified_name516919:&"
 
_user_specified_name516917:&"
 
_user_specified_name516915:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
I
-__inference_activation_1_layer_call_fn_516997

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_515505d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
S
'__inference_add_14_layer_call_fn_516841
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_add_14_layer_call_and_return_conditional_losses_515433d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������>:���������>:UQ
+
_output_shapes
:���������>
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������>
"
_user_specified_name
inputs_0
�'
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_515062

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
7__inference_layer_normalization_14_layer_call_fn_516856

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_layer_normalization_14_layer_call_and_return_conditional_losses_515456s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name516852:&"
 
_user_specified_name516850:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
c
*__inference_dropout_1_layer_call_fn_517007

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_515518s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
R__inference_layer_normalization_14_layer_call_and_return_conditional_losses_516878

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������>�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������>l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������>a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������>~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������>f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������>\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
D__inference_dense_18_layer_call_and_return_conditional_losses_516515

inputs3
!tensordot_readvariableop_resource:
identity��Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������>�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������>e
IdentityIdentityTensordot:output:0^NoOp*
T0*+
_output_shapes
:���������>=
NoOpNoOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������>: 24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�4
�
R__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_515410	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:͓>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������>�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������>>*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������>>Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������>>l
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
::���
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������>>*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������>>\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������>>�
einsum_1/EinsumEinsum!dropout/dropout/SelectV2:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������>*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������>*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:���������>�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������>:���������>: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������>

_user_specified_namevalue:R N
+
_output_shapes
:���������>

_user_specified_namequery
�
�
D__inference_dense_21_layer_call_and_return_conditional_losses_515549

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������>�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������>r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������>V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
��
�x
__inference__traced_save_517873
file_prefixA
3read_disablecopyonread_layer_normalization_12_gamma:B
4read_1_disablecopyonread_layer_normalization_12_beta::
(read_2_disablecopyonread_dense_18_kernel:@
2read_3_disablecopyonread_batch_normalization_gamma:?
1read_4_disablecopyonread_batch_normalization_beta:F
8read_5_disablecopyonread_batch_normalization_moving_mean:J
<read_6_disablecopyonread_batch_normalization_moving_variance::
(read_7_disablecopyonread_dense_19_kernel:4
&read_8_disablecopyonread_dense_19_bias:C
5read_9_disablecopyonread_layer_normalization_13_gamma:C
5read_10_disablecopyonread_layer_normalization_13_beta:D
6read_11_disablecopyonread_layer_normalization_14_gamma:C
5read_12_disablecopyonread_layer_normalization_14_beta:;
)read_13_disablecopyonread_dense_20_kernel:C
5read_14_disablecopyonread_batch_normalization_1_gamma:B
4read_15_disablecopyonread_batch_normalization_1_beta:I
;read_16_disablecopyonread_batch_normalization_1_moving_mean:M
?read_17_disablecopyonread_batch_normalization_1_moving_variance:;
)read_18_disablecopyonread_dense_21_kernel:5
'read_19_disablecopyonread_dense_21_bias:D
6read_20_disablecopyonread_layer_normalization_15_gamma:C
5read_21_disablecopyonread_layer_normalization_15_beta:;
)read_22_disablecopyonread_dense_22_kernel:5
'read_23_disablecopyonread_dense_22_bias:S
=read_24_disablecopyonread_multi_head_attention_6_query_kernel:M
;read_25_disablecopyonread_multi_head_attention_6_query_bias:Q
;read_26_disablecopyonread_multi_head_attention_6_key_kernel:K
9read_27_disablecopyonread_multi_head_attention_6_key_bias:S
=read_28_disablecopyonread_multi_head_attention_6_value_kernel:M
;read_29_disablecopyonread_multi_head_attention_6_value_bias:^
Hread_30_disablecopyonread_multi_head_attention_6_attention_output_kernel:T
Fread_31_disablecopyonread_multi_head_attention_6_attention_output_bias:S
=read_32_disablecopyonread_multi_head_attention_7_query_kernel:M
;read_33_disablecopyonread_multi_head_attention_7_query_bias:Q
;read_34_disablecopyonread_multi_head_attention_7_key_kernel:K
9read_35_disablecopyonread_multi_head_attention_7_key_bias:S
=read_36_disablecopyonread_multi_head_attention_7_value_kernel:M
;read_37_disablecopyonread_multi_head_attention_7_value_bias:^
Hread_38_disablecopyonread_multi_head_attention_7_attention_output_kernel:T
Fread_39_disablecopyonread_multi_head_attention_7_attention_output_bias:-
#read_40_disablecopyonread_iteration:	 1
'read_41_disablecopyonread_learning_rate: Z
Dread_42_disablecopyonread_adam_m_multi_head_attention_6_query_kernel:Z
Dread_43_disablecopyonread_adam_v_multi_head_attention_6_query_kernel:T
Bread_44_disablecopyonread_adam_m_multi_head_attention_6_query_bias:T
Bread_45_disablecopyonread_adam_v_multi_head_attention_6_query_bias:X
Bread_46_disablecopyonread_adam_m_multi_head_attention_6_key_kernel:X
Bread_47_disablecopyonread_adam_v_multi_head_attention_6_key_kernel:R
@read_48_disablecopyonread_adam_m_multi_head_attention_6_key_bias:R
@read_49_disablecopyonread_adam_v_multi_head_attention_6_key_bias:Z
Dread_50_disablecopyonread_adam_m_multi_head_attention_6_value_kernel:Z
Dread_51_disablecopyonread_adam_v_multi_head_attention_6_value_kernel:T
Bread_52_disablecopyonread_adam_m_multi_head_attention_6_value_bias:T
Bread_53_disablecopyonread_adam_v_multi_head_attention_6_value_bias:e
Oread_54_disablecopyonread_adam_m_multi_head_attention_6_attention_output_kernel:e
Oread_55_disablecopyonread_adam_v_multi_head_attention_6_attention_output_kernel:[
Mread_56_disablecopyonread_adam_m_multi_head_attention_6_attention_output_bias:[
Mread_57_disablecopyonread_adam_v_multi_head_attention_6_attention_output_bias:K
=read_58_disablecopyonread_adam_m_layer_normalization_12_gamma:K
=read_59_disablecopyonread_adam_v_layer_normalization_12_gamma:J
<read_60_disablecopyonread_adam_m_layer_normalization_12_beta:J
<read_61_disablecopyonread_adam_v_layer_normalization_12_beta:B
0read_62_disablecopyonread_adam_m_dense_18_kernel:B
0read_63_disablecopyonread_adam_v_dense_18_kernel:H
:read_64_disablecopyonread_adam_m_batch_normalization_gamma:H
:read_65_disablecopyonread_adam_v_batch_normalization_gamma:G
9read_66_disablecopyonread_adam_m_batch_normalization_beta:G
9read_67_disablecopyonread_adam_v_batch_normalization_beta:B
0read_68_disablecopyonread_adam_m_dense_19_kernel:B
0read_69_disablecopyonread_adam_v_dense_19_kernel:<
.read_70_disablecopyonread_adam_m_dense_19_bias:<
.read_71_disablecopyonread_adam_v_dense_19_bias:K
=read_72_disablecopyonread_adam_m_layer_normalization_13_gamma:K
=read_73_disablecopyonread_adam_v_layer_normalization_13_gamma:J
<read_74_disablecopyonread_adam_m_layer_normalization_13_beta:J
<read_75_disablecopyonread_adam_v_layer_normalization_13_beta:Z
Dread_76_disablecopyonread_adam_m_multi_head_attention_7_query_kernel:Z
Dread_77_disablecopyonread_adam_v_multi_head_attention_7_query_kernel:T
Bread_78_disablecopyonread_adam_m_multi_head_attention_7_query_bias:T
Bread_79_disablecopyonread_adam_v_multi_head_attention_7_query_bias:X
Bread_80_disablecopyonread_adam_m_multi_head_attention_7_key_kernel:X
Bread_81_disablecopyonread_adam_v_multi_head_attention_7_key_kernel:R
@read_82_disablecopyonread_adam_m_multi_head_attention_7_key_bias:R
@read_83_disablecopyonread_adam_v_multi_head_attention_7_key_bias:Z
Dread_84_disablecopyonread_adam_m_multi_head_attention_7_value_kernel:Z
Dread_85_disablecopyonread_adam_v_multi_head_attention_7_value_kernel:T
Bread_86_disablecopyonread_adam_m_multi_head_attention_7_value_bias:T
Bread_87_disablecopyonread_adam_v_multi_head_attention_7_value_bias:e
Oread_88_disablecopyonread_adam_m_multi_head_attention_7_attention_output_kernel:e
Oread_89_disablecopyonread_adam_v_multi_head_attention_7_attention_output_kernel:[
Mread_90_disablecopyonread_adam_m_multi_head_attention_7_attention_output_bias:[
Mread_91_disablecopyonread_adam_v_multi_head_attention_7_attention_output_bias:K
=read_92_disablecopyonread_adam_m_layer_normalization_14_gamma:K
=read_93_disablecopyonread_adam_v_layer_normalization_14_gamma:J
<read_94_disablecopyonread_adam_m_layer_normalization_14_beta:J
<read_95_disablecopyonread_adam_v_layer_normalization_14_beta:B
0read_96_disablecopyonread_adam_m_dense_20_kernel:B
0read_97_disablecopyonread_adam_v_dense_20_kernel:J
<read_98_disablecopyonread_adam_m_batch_normalization_1_gamma:J
<read_99_disablecopyonread_adam_v_batch_normalization_1_gamma:J
<read_100_disablecopyonread_adam_m_batch_normalization_1_beta:J
<read_101_disablecopyonread_adam_v_batch_normalization_1_beta:C
1read_102_disablecopyonread_adam_m_dense_21_kernel:C
1read_103_disablecopyonread_adam_v_dense_21_kernel:=
/read_104_disablecopyonread_adam_m_dense_21_bias:=
/read_105_disablecopyonread_adam_v_dense_21_bias:L
>read_106_disablecopyonread_adam_m_layer_normalization_15_gamma:L
>read_107_disablecopyonread_adam_v_layer_normalization_15_gamma:K
=read_108_disablecopyonread_adam_m_layer_normalization_15_beta:K
=read_109_disablecopyonread_adam_v_layer_normalization_15_beta:C
1read_110_disablecopyonread_adam_m_dense_22_kernel:C
1read_111_disablecopyonread_adam_v_dense_22_kernel:=
/read_112_disablecopyonread_adam_m_dense_22_bias:=
/read_113_disablecopyonread_adam_v_dense_22_bias:,
"read_114_disablecopyonread_total_1: ,
"read_115_disablecopyonread_count_1: *
 read_116_disablecopyonread_total: *
 read_117_disablecopyonread_count: 
savev2_const_1
identity_237��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_100/DisableCopyOnRead�Read_100/ReadVariableOp�Read_101/DisableCopyOnRead�Read_101/ReadVariableOp�Read_102/DisableCopyOnRead�Read_102/ReadVariableOp�Read_103/DisableCopyOnRead�Read_103/ReadVariableOp�Read_104/DisableCopyOnRead�Read_104/ReadVariableOp�Read_105/DisableCopyOnRead�Read_105/ReadVariableOp�Read_106/DisableCopyOnRead�Read_106/ReadVariableOp�Read_107/DisableCopyOnRead�Read_107/ReadVariableOp�Read_108/DisableCopyOnRead�Read_108/ReadVariableOp�Read_109/DisableCopyOnRead�Read_109/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_110/DisableCopyOnRead�Read_110/ReadVariableOp�Read_111/DisableCopyOnRead�Read_111/ReadVariableOp�Read_112/DisableCopyOnRead�Read_112/ReadVariableOp�Read_113/DisableCopyOnRead�Read_113/ReadVariableOp�Read_114/DisableCopyOnRead�Read_114/ReadVariableOp�Read_115/DisableCopyOnRead�Read_115/ReadVariableOp�Read_116/DisableCopyOnRead�Read_116/ReadVariableOp�Read_117/DisableCopyOnRead�Read_117/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_80/DisableCopyOnRead�Read_80/ReadVariableOp�Read_81/DisableCopyOnRead�Read_81/ReadVariableOp�Read_82/DisableCopyOnRead�Read_82/ReadVariableOp�Read_83/DisableCopyOnRead�Read_83/ReadVariableOp�Read_84/DisableCopyOnRead�Read_84/ReadVariableOp�Read_85/DisableCopyOnRead�Read_85/ReadVariableOp�Read_86/DisableCopyOnRead�Read_86/ReadVariableOp�Read_87/DisableCopyOnRead�Read_87/ReadVariableOp�Read_88/DisableCopyOnRead�Read_88/ReadVariableOp�Read_89/DisableCopyOnRead�Read_89/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOp�Read_90/DisableCopyOnRead�Read_90/ReadVariableOp�Read_91/DisableCopyOnRead�Read_91/ReadVariableOp�Read_92/DisableCopyOnRead�Read_92/ReadVariableOp�Read_93/DisableCopyOnRead�Read_93/ReadVariableOp�Read_94/DisableCopyOnRead�Read_94/ReadVariableOp�Read_95/DisableCopyOnRead�Read_95/ReadVariableOp�Read_96/DisableCopyOnRead�Read_96/ReadVariableOp�Read_97/DisableCopyOnRead�Read_97/ReadVariableOp�Read_98/DisableCopyOnRead�Read_98/ReadVariableOp�Read_99/DisableCopyOnRead�Read_99/ReadVariableOpw
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
: �
Read/DisableCopyOnReadDisableCopyOnRead3read_disablecopyonread_layer_normalization_12_gamma"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp3read_disablecopyonread_layer_normalization_12_gamma^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0e
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:]

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_1/DisableCopyOnReadDisableCopyOnRead4read_1_disablecopyonread_layer_normalization_12_beta"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp4read_1_disablecopyonread_layer_normalization_12_beta^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_18_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_18_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_3/DisableCopyOnReadDisableCopyOnRead2read_3_disablecopyonread_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp2read_3_disablecopyonread_batch_normalization_gamma^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_4/DisableCopyOnReadDisableCopyOnRead1read_4_disablecopyonread_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp1read_4_disablecopyonread_batch_normalization_beta^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_5/DisableCopyOnReadDisableCopyOnRead8read_5_disablecopyonread_batch_normalization_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp8read_5_disablecopyonread_batch_normalization_moving_mean^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_6/DisableCopyOnReadDisableCopyOnRead<read_6_disablecopyonread_batch_normalization_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp<read_6_disablecopyonread_batch_normalization_moving_variance^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_7/DisableCopyOnReadDisableCopyOnRead(read_7_disablecopyonread_dense_19_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp(read_7_disablecopyonread_dense_19_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_8/DisableCopyOnReadDisableCopyOnRead&read_8_disablecopyonread_dense_19_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp&read_8_disablecopyonread_dense_19_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_9/DisableCopyOnReadDisableCopyOnRead5read_9_disablecopyonread_layer_normalization_13_gamma"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp5read_9_disablecopyonread_layer_normalization_13_gamma^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_10/DisableCopyOnReadDisableCopyOnRead5read_10_disablecopyonread_layer_normalization_13_beta"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp5read_10_disablecopyonread_layer_normalization_13_beta^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_11/DisableCopyOnReadDisableCopyOnRead6read_11_disablecopyonread_layer_normalization_14_gamma"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp6read_11_disablecopyonread_layer_normalization_14_gamma^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_12/DisableCopyOnReadDisableCopyOnRead5read_12_disablecopyonread_layer_normalization_14_beta"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp5read_12_disablecopyonread_layer_normalization_14_beta^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_13/DisableCopyOnReadDisableCopyOnRead)read_13_disablecopyonread_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp)read_13_disablecopyonread_dense_20_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_14/DisableCopyOnReadDisableCopyOnRead5read_14_disablecopyonread_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp5read_14_disablecopyonread_batch_normalization_1_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_15/DisableCopyOnReadDisableCopyOnRead4read_15_disablecopyonread_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp4read_15_disablecopyonread_batch_normalization_1_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnRead;read_16_disablecopyonread_batch_normalization_1_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp;read_16_disablecopyonread_batch_normalization_1_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_17/DisableCopyOnReadDisableCopyOnRead?read_17_disablecopyonread_batch_normalization_1_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp?read_17_disablecopyonread_batch_normalization_1_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_18/DisableCopyOnReadDisableCopyOnRead)read_18_disablecopyonread_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp)read_18_disablecopyonread_dense_21_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

:|
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_dense_21_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_20/DisableCopyOnReadDisableCopyOnRead6read_20_disablecopyonread_layer_normalization_15_gamma"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp6read_20_disablecopyonread_layer_normalization_15_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_21/DisableCopyOnReadDisableCopyOnRead5read_21_disablecopyonread_layer_normalization_15_beta"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp5read_21_disablecopyonread_layer_normalization_15_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_22/DisableCopyOnReadDisableCopyOnRead)read_22_disablecopyonread_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp)read_22_disablecopyonread_dense_22_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:|
Read_23/DisableCopyOnReadDisableCopyOnRead'read_23_disablecopyonread_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp'read_23_disablecopyonread_dense_22_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_24/DisableCopyOnReadDisableCopyOnRead=read_24_disablecopyonread_multi_head_attention_6_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp=read_24_disablecopyonread_multi_head_attention_6_query_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_25/DisableCopyOnReadDisableCopyOnRead;read_25_disablecopyonread_multi_head_attention_6_query_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp;read_25_disablecopyonread_multi_head_attention_6_query_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_26/DisableCopyOnReadDisableCopyOnRead;read_26_disablecopyonread_multi_head_attention_6_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp;read_26_disablecopyonread_multi_head_attention_6_key_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_27/DisableCopyOnReadDisableCopyOnRead9read_27_disablecopyonread_multi_head_attention_6_key_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp9read_27_disablecopyonread_multi_head_attention_6_key_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_28/DisableCopyOnReadDisableCopyOnRead=read_28_disablecopyonread_multi_head_attention_6_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp=read_28_disablecopyonread_multi_head_attention_6_value_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_29/DisableCopyOnReadDisableCopyOnRead;read_29_disablecopyonread_multi_head_attention_6_value_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp;read_29_disablecopyonread_multi_head_attention_6_value_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_30/DisableCopyOnReadDisableCopyOnReadHread_30_disablecopyonread_multi_head_attention_6_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOpHread_30_disablecopyonread_multi_head_attention_6_attention_output_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_31/DisableCopyOnReadDisableCopyOnReadFread_31_disablecopyonread_multi_head_attention_6_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOpFread_31_disablecopyonread_multi_head_attention_6_attention_output_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_32/DisableCopyOnReadDisableCopyOnRead=read_32_disablecopyonread_multi_head_attention_7_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp=read_32_disablecopyonread_multi_head_attention_7_query_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_33/DisableCopyOnReadDisableCopyOnRead;read_33_disablecopyonread_multi_head_attention_7_query_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp;read_33_disablecopyonread_multi_head_attention_7_query_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_34/DisableCopyOnReadDisableCopyOnRead;read_34_disablecopyonread_multi_head_attention_7_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp;read_34_disablecopyonread_multi_head_attention_7_key_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_35/DisableCopyOnReadDisableCopyOnRead9read_35_disablecopyonread_multi_head_attention_7_key_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp9read_35_disablecopyonread_multi_head_attention_7_key_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_36/DisableCopyOnReadDisableCopyOnRead=read_36_disablecopyonread_multi_head_attention_7_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp=read_36_disablecopyonread_multi_head_attention_7_value_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_37/DisableCopyOnReadDisableCopyOnRead;read_37_disablecopyonread_multi_head_attention_7_value_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp;read_37_disablecopyonread_multi_head_attention_7_value_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_38/DisableCopyOnReadDisableCopyOnReadHread_38_disablecopyonread_multi_head_attention_7_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOpHread_38_disablecopyonread_multi_head_attention_7_attention_output_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_39/DisableCopyOnReadDisableCopyOnReadFread_39_disablecopyonread_multi_head_attention_7_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOpFread_39_disablecopyonread_multi_head_attention_7_attention_output_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_40/DisableCopyOnReadDisableCopyOnRead#read_40_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp#read_40_disablecopyonread_iteration^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_41/DisableCopyOnReadDisableCopyOnRead'read_41_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp'read_41_disablecopyonread_learning_rate^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_42/DisableCopyOnReadDisableCopyOnReadDread_42_disablecopyonread_adam_m_multi_head_attention_6_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOpDread_42_disablecopyonread_adam_m_multi_head_attention_6_query_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_43/DisableCopyOnReadDisableCopyOnReadDread_43_disablecopyonread_adam_v_multi_head_attention_6_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOpDread_43_disablecopyonread_adam_v_multi_head_attention_6_query_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_44/DisableCopyOnReadDisableCopyOnReadBread_44_disablecopyonread_adam_m_multi_head_attention_6_query_bias"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOpBread_44_disablecopyonread_adam_m_multi_head_attention_6_query_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_45/DisableCopyOnReadDisableCopyOnReadBread_45_disablecopyonread_adam_v_multi_head_attention_6_query_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOpBread_45_disablecopyonread_adam_v_multi_head_attention_6_query_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_46/DisableCopyOnReadDisableCopyOnReadBread_46_disablecopyonread_adam_m_multi_head_attention_6_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOpBread_46_disablecopyonread_adam_m_multi_head_attention_6_key_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_47/DisableCopyOnReadDisableCopyOnReadBread_47_disablecopyonread_adam_v_multi_head_attention_6_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOpBread_47_disablecopyonread_adam_v_multi_head_attention_6_key_kernel^Read_47/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_48/DisableCopyOnReadDisableCopyOnRead@read_48_disablecopyonread_adam_m_multi_head_attention_6_key_bias"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp@read_48_disablecopyonread_adam_m_multi_head_attention_6_key_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_49/DisableCopyOnReadDisableCopyOnRead@read_49_disablecopyonread_adam_v_multi_head_attention_6_key_bias"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp@read_49_disablecopyonread_adam_v_multi_head_attention_6_key_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_50/DisableCopyOnReadDisableCopyOnReadDread_50_disablecopyonread_adam_m_multi_head_attention_6_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOpDread_50_disablecopyonread_adam_m_multi_head_attention_6_value_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_51/DisableCopyOnReadDisableCopyOnReadDread_51_disablecopyonread_adam_v_multi_head_attention_6_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOpDread_51_disablecopyonread_adam_v_multi_head_attention_6_value_kernel^Read_51/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_52/DisableCopyOnReadDisableCopyOnReadBread_52_disablecopyonread_adam_m_multi_head_attention_6_value_bias"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOpBread_52_disablecopyonread_adam_m_multi_head_attention_6_value_bias^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_53/DisableCopyOnReadDisableCopyOnReadBread_53_disablecopyonread_adam_v_multi_head_attention_6_value_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpBread_53_disablecopyonread_adam_v_multi_head_attention_6_value_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_54/DisableCopyOnReadDisableCopyOnReadOread_54_disablecopyonread_adam_m_multi_head_attention_6_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOpOread_54_disablecopyonread_adam_m_multi_head_attention_6_attention_output_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_55/DisableCopyOnReadDisableCopyOnReadOread_55_disablecopyonread_adam_v_multi_head_attention_6_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOpOread_55_disablecopyonread_adam_v_multi_head_attention_6_attention_output_kernel^Read_55/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_56/DisableCopyOnReadDisableCopyOnReadMread_56_disablecopyonread_adam_m_multi_head_attention_6_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOpMread_56_disablecopyonread_adam_m_multi_head_attention_6_attention_output_bias^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_57/DisableCopyOnReadDisableCopyOnReadMread_57_disablecopyonread_adam_v_multi_head_attention_6_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOpMread_57_disablecopyonread_adam_v_multi_head_attention_6_attention_output_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_58/DisableCopyOnReadDisableCopyOnRead=read_58_disablecopyonread_adam_m_layer_normalization_12_gamma"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp=read_58_disablecopyonread_adam_m_layer_normalization_12_gamma^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_59/DisableCopyOnReadDisableCopyOnRead=read_59_disablecopyonread_adam_v_layer_normalization_12_gamma"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp=read_59_disablecopyonread_adam_v_layer_normalization_12_gamma^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_60/DisableCopyOnReadDisableCopyOnRead<read_60_disablecopyonread_adam_m_layer_normalization_12_beta"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp<read_60_disablecopyonread_adam_m_layer_normalization_12_beta^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_61/DisableCopyOnReadDisableCopyOnRead<read_61_disablecopyonread_adam_v_layer_normalization_12_beta"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp<read_61_disablecopyonread_adam_v_layer_normalization_12_beta^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_62/DisableCopyOnReadDisableCopyOnRead0read_62_disablecopyonread_adam_m_dense_18_kernel"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp0read_62_disablecopyonread_adam_m_dense_18_kernel^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_63/DisableCopyOnReadDisableCopyOnRead0read_63_disablecopyonread_adam_v_dense_18_kernel"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp0read_63_disablecopyonread_adam_v_dense_18_kernel^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_64/DisableCopyOnReadDisableCopyOnRead:read_64_disablecopyonread_adam_m_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp:read_64_disablecopyonread_adam_m_batch_normalization_gamma^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_65/DisableCopyOnReadDisableCopyOnRead:read_65_disablecopyonread_adam_v_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp:read_65_disablecopyonread_adam_v_batch_normalization_gamma^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_66/DisableCopyOnReadDisableCopyOnRead9read_66_disablecopyonread_adam_m_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp9read_66_disablecopyonread_adam_m_batch_normalization_beta^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_67/DisableCopyOnReadDisableCopyOnRead9read_67_disablecopyonread_adam_v_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp9read_67_disablecopyonread_adam_v_batch_normalization_beta^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_68/DisableCopyOnReadDisableCopyOnRead0read_68_disablecopyonread_adam_m_dense_19_kernel"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp0read_68_disablecopyonread_adam_m_dense_19_kernel^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_69/DisableCopyOnReadDisableCopyOnRead0read_69_disablecopyonread_adam_v_dense_19_kernel"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp0read_69_disablecopyonread_adam_v_dense_19_kernel^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_70/DisableCopyOnReadDisableCopyOnRead.read_70_disablecopyonread_adam_m_dense_19_bias"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp.read_70_disablecopyonread_adam_m_dense_19_bias^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_71/DisableCopyOnReadDisableCopyOnRead.read_71_disablecopyonread_adam_v_dense_19_bias"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp.read_71_disablecopyonread_adam_v_dense_19_bias^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_72/DisableCopyOnReadDisableCopyOnRead=read_72_disablecopyonread_adam_m_layer_normalization_13_gamma"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp=read_72_disablecopyonread_adam_m_layer_normalization_13_gamma^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_73/DisableCopyOnReadDisableCopyOnRead=read_73_disablecopyonread_adam_v_layer_normalization_13_gamma"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp=read_73_disablecopyonread_adam_v_layer_normalization_13_gamma^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_74/DisableCopyOnReadDisableCopyOnRead<read_74_disablecopyonread_adam_m_layer_normalization_13_beta"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOp<read_74_disablecopyonread_adam_m_layer_normalization_13_beta^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_75/DisableCopyOnReadDisableCopyOnRead<read_75_disablecopyonread_adam_v_layer_normalization_13_beta"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOp<read_75_disablecopyonread_adam_v_layer_normalization_13_beta^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_76/DisableCopyOnReadDisableCopyOnReadDread_76_disablecopyonread_adam_m_multi_head_attention_7_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOpDread_76_disablecopyonread_adam_m_multi_head_attention_7_query_kernel^Read_76/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_77/DisableCopyOnReadDisableCopyOnReadDread_77_disablecopyonread_adam_v_multi_head_attention_7_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOpDread_77_disablecopyonread_adam_v_multi_head_attention_7_query_kernel^Read_77/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_78/DisableCopyOnReadDisableCopyOnReadBread_78_disablecopyonread_adam_m_multi_head_attention_7_query_bias"/device:CPU:0*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOpBread_78_disablecopyonread_adam_m_multi_head_attention_7_query_bias^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_79/DisableCopyOnReadDisableCopyOnReadBread_79_disablecopyonread_adam_v_multi_head_attention_7_query_bias"/device:CPU:0*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOpBread_79_disablecopyonread_adam_v_multi_head_attention_7_query_bias^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_80/DisableCopyOnReadDisableCopyOnReadBread_80_disablecopyonread_adam_m_multi_head_attention_7_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_80/ReadVariableOpReadVariableOpBread_80_disablecopyonread_adam_m_multi_head_attention_7_key_kernel^Read_80/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_81/DisableCopyOnReadDisableCopyOnReadBread_81_disablecopyonread_adam_v_multi_head_attention_7_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_81/ReadVariableOpReadVariableOpBread_81_disablecopyonread_adam_v_multi_head_attention_7_key_kernel^Read_81/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_82/DisableCopyOnReadDisableCopyOnRead@read_82_disablecopyonread_adam_m_multi_head_attention_7_key_bias"/device:CPU:0*
_output_shapes
 �
Read_82/ReadVariableOpReadVariableOp@read_82_disablecopyonread_adam_m_multi_head_attention_7_key_bias^Read_82/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_83/DisableCopyOnReadDisableCopyOnRead@read_83_disablecopyonread_adam_v_multi_head_attention_7_key_bias"/device:CPU:0*
_output_shapes
 �
Read_83/ReadVariableOpReadVariableOp@read_83_disablecopyonread_adam_v_multi_head_attention_7_key_bias^Read_83/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_84/DisableCopyOnReadDisableCopyOnReadDread_84_disablecopyonread_adam_m_multi_head_attention_7_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_84/ReadVariableOpReadVariableOpDread_84_disablecopyonread_adam_m_multi_head_attention_7_value_kernel^Read_84/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_85/DisableCopyOnReadDisableCopyOnReadDread_85_disablecopyonread_adam_v_multi_head_attention_7_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_85/ReadVariableOpReadVariableOpDread_85_disablecopyonread_adam_v_multi_head_attention_7_value_kernel^Read_85/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_86/DisableCopyOnReadDisableCopyOnReadBread_86_disablecopyonread_adam_m_multi_head_attention_7_value_bias"/device:CPU:0*
_output_shapes
 �
Read_86/ReadVariableOpReadVariableOpBread_86_disablecopyonread_adam_m_multi_head_attention_7_value_bias^Read_86/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_87/DisableCopyOnReadDisableCopyOnReadBread_87_disablecopyonread_adam_v_multi_head_attention_7_value_bias"/device:CPU:0*
_output_shapes
 �
Read_87/ReadVariableOpReadVariableOpBread_87_disablecopyonread_adam_v_multi_head_attention_7_value_bias^Read_87/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_88/DisableCopyOnReadDisableCopyOnReadOread_88_disablecopyonread_adam_m_multi_head_attention_7_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_88/ReadVariableOpReadVariableOpOread_88_disablecopyonread_adam_m_multi_head_attention_7_attention_output_kernel^Read_88/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_89/DisableCopyOnReadDisableCopyOnReadOread_89_disablecopyonread_adam_v_multi_head_attention_7_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_89/ReadVariableOpReadVariableOpOread_89_disablecopyonread_adam_v_multi_head_attention_7_attention_output_kernel^Read_89/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0t
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:k
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_90/DisableCopyOnReadDisableCopyOnReadMread_90_disablecopyonread_adam_m_multi_head_attention_7_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_90/ReadVariableOpReadVariableOpMread_90_disablecopyonread_adam_m_multi_head_attention_7_attention_output_bias^Read_90/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_91/DisableCopyOnReadDisableCopyOnReadMread_91_disablecopyonread_adam_v_multi_head_attention_7_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_91/ReadVariableOpReadVariableOpMread_91_disablecopyonread_adam_v_multi_head_attention_7_attention_output_bias^Read_91/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_92/DisableCopyOnReadDisableCopyOnRead=read_92_disablecopyonread_adam_m_layer_normalization_14_gamma"/device:CPU:0*
_output_shapes
 �
Read_92/ReadVariableOpReadVariableOp=read_92_disablecopyonread_adam_m_layer_normalization_14_gamma^Read_92/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_93/DisableCopyOnReadDisableCopyOnRead=read_93_disablecopyonread_adam_v_layer_normalization_14_gamma"/device:CPU:0*
_output_shapes
 �
Read_93/ReadVariableOpReadVariableOp=read_93_disablecopyonread_adam_v_layer_normalization_14_gamma^Read_93/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_94/DisableCopyOnReadDisableCopyOnRead<read_94_disablecopyonread_adam_m_layer_normalization_14_beta"/device:CPU:0*
_output_shapes
 �
Read_94/ReadVariableOpReadVariableOp<read_94_disablecopyonread_adam_m_layer_normalization_14_beta^Read_94/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_95/DisableCopyOnReadDisableCopyOnRead<read_95_disablecopyonread_adam_v_layer_normalization_14_beta"/device:CPU:0*
_output_shapes
 �
Read_95/ReadVariableOpReadVariableOp<read_95_disablecopyonread_adam_v_layer_normalization_14_beta^Read_95/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_96/DisableCopyOnReadDisableCopyOnRead0read_96_disablecopyonread_adam_m_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_96/ReadVariableOpReadVariableOp0read_96_disablecopyonread_adam_m_dense_20_kernel^Read_96/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_192IdentityRead_96/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_193IdentityIdentity_192:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_97/DisableCopyOnReadDisableCopyOnRead0read_97_disablecopyonread_adam_v_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_97/ReadVariableOpReadVariableOp0read_97_disablecopyonread_adam_v_dense_20_kernel^Read_97/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_194IdentityRead_97/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_195IdentityIdentity_194:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_98/DisableCopyOnReadDisableCopyOnRead<read_98_disablecopyonread_adam_m_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_98/ReadVariableOpReadVariableOp<read_98_disablecopyonread_adam_m_batch_normalization_1_gamma^Read_98/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_196IdentityRead_98/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_197IdentityIdentity_196:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_99/DisableCopyOnReadDisableCopyOnRead<read_99_disablecopyonread_adam_v_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_99/ReadVariableOpReadVariableOp<read_99_disablecopyonread_adam_v_batch_normalization_1_gamma^Read_99/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_198IdentityRead_99/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_199IdentityIdentity_198:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_100/DisableCopyOnReadDisableCopyOnRead<read_100_disablecopyonread_adam_m_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_100/ReadVariableOpReadVariableOp<read_100_disablecopyonread_adam_m_batch_normalization_1_beta^Read_100/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_200IdentityRead_100/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_201IdentityIdentity_200:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_101/DisableCopyOnReadDisableCopyOnRead<read_101_disablecopyonread_adam_v_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_101/ReadVariableOpReadVariableOp<read_101_disablecopyonread_adam_v_batch_normalization_1_beta^Read_101/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_202IdentityRead_101/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_203IdentityIdentity_202:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_102/DisableCopyOnReadDisableCopyOnRead1read_102_disablecopyonread_adam_m_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_102/ReadVariableOpReadVariableOp1read_102_disablecopyonread_adam_m_dense_21_kernel^Read_102/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0q
Identity_204IdentityRead_102/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_205IdentityIdentity_204:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_103/DisableCopyOnReadDisableCopyOnRead1read_103_disablecopyonread_adam_v_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_103/ReadVariableOpReadVariableOp1read_103_disablecopyonread_adam_v_dense_21_kernel^Read_103/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0q
Identity_206IdentityRead_103/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_207IdentityIdentity_206:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_104/DisableCopyOnReadDisableCopyOnRead/read_104_disablecopyonread_adam_m_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_104/ReadVariableOpReadVariableOp/read_104_disablecopyonread_adam_m_dense_21_bias^Read_104/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_208IdentityRead_104/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_209IdentityIdentity_208:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_105/DisableCopyOnReadDisableCopyOnRead/read_105_disablecopyonread_adam_v_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_105/ReadVariableOpReadVariableOp/read_105_disablecopyonread_adam_v_dense_21_bias^Read_105/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_210IdentityRead_105/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_211IdentityIdentity_210:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_106/DisableCopyOnReadDisableCopyOnRead>read_106_disablecopyonread_adam_m_layer_normalization_15_gamma"/device:CPU:0*
_output_shapes
 �
Read_106/ReadVariableOpReadVariableOp>read_106_disablecopyonread_adam_m_layer_normalization_15_gamma^Read_106/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_212IdentityRead_106/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_213IdentityIdentity_212:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_107/DisableCopyOnReadDisableCopyOnRead>read_107_disablecopyonread_adam_v_layer_normalization_15_gamma"/device:CPU:0*
_output_shapes
 �
Read_107/ReadVariableOpReadVariableOp>read_107_disablecopyonread_adam_v_layer_normalization_15_gamma^Read_107/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_214IdentityRead_107/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_215IdentityIdentity_214:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_108/DisableCopyOnReadDisableCopyOnRead=read_108_disablecopyonread_adam_m_layer_normalization_15_beta"/device:CPU:0*
_output_shapes
 �
Read_108/ReadVariableOpReadVariableOp=read_108_disablecopyonread_adam_m_layer_normalization_15_beta^Read_108/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_216IdentityRead_108/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_217IdentityIdentity_216:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_109/DisableCopyOnReadDisableCopyOnRead=read_109_disablecopyonread_adam_v_layer_normalization_15_beta"/device:CPU:0*
_output_shapes
 �
Read_109/ReadVariableOpReadVariableOp=read_109_disablecopyonread_adam_v_layer_normalization_15_beta^Read_109/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_218IdentityRead_109/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_219IdentityIdentity_218:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_110/DisableCopyOnReadDisableCopyOnRead1read_110_disablecopyonread_adam_m_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_110/ReadVariableOpReadVariableOp1read_110_disablecopyonread_adam_m_dense_22_kernel^Read_110/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0q
Identity_220IdentityRead_110/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_221IdentityIdentity_220:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_111/DisableCopyOnReadDisableCopyOnRead1read_111_disablecopyonread_adam_v_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_111/ReadVariableOpReadVariableOp1read_111_disablecopyonread_adam_v_dense_22_kernel^Read_111/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0q
Identity_222IdentityRead_111/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_223IdentityIdentity_222:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_112/DisableCopyOnReadDisableCopyOnRead/read_112_disablecopyonread_adam_m_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_112/ReadVariableOpReadVariableOp/read_112_disablecopyonread_adam_m_dense_22_bias^Read_112/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_224IdentityRead_112/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_225IdentityIdentity_224:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_113/DisableCopyOnReadDisableCopyOnRead/read_113_disablecopyonread_adam_v_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_113/ReadVariableOpReadVariableOp/read_113_disablecopyonread_adam_v_dense_22_bias^Read_113/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_226IdentityRead_113/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_227IdentityIdentity_226:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_114/DisableCopyOnReadDisableCopyOnRead"read_114_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_114/ReadVariableOpReadVariableOp"read_114_disablecopyonread_total_1^Read_114/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_228IdentityRead_114/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_229IdentityIdentity_228:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_115/DisableCopyOnReadDisableCopyOnRead"read_115_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_115/ReadVariableOpReadVariableOp"read_115_disablecopyonread_count_1^Read_115/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_230IdentityRead_115/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_231IdentityIdentity_230:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_116/DisableCopyOnReadDisableCopyOnRead read_116_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_116/ReadVariableOpReadVariableOp read_116_disablecopyonread_total^Read_116/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_232IdentityRead_116/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_233IdentityIdentity_232:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_117/DisableCopyOnReadDisableCopyOnRead read_117_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_117/ReadVariableOpReadVariableOp read_117_disablecopyonread_count^Read_117/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_234IdentityRead_117/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_235IdentityIdentity_234:output:0"/device:CPU:0*
T0*
_output_shapes
: �0
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:w*
dtype0*�/
value�/B�/wB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:w*
dtype0*�
value�B�wB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0Identity_193:output:0Identity_195:output:0Identity_197:output:0Identity_199:output:0Identity_201:output:0Identity_203:output:0Identity_205:output:0Identity_207:output:0Identity_209:output:0Identity_211:output:0Identity_213:output:0Identity_215:output:0Identity_217:output:0Identity_219:output:0Identity_221:output:0Identity_223:output:0Identity_225:output:0Identity_227:output:0Identity_229:output:0Identity_231:output:0Identity_233:output:0Identity_235:output:0savev2_const_1"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *�
dtypes{
y2w	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_236Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_237IdentityIdentity_236:output:0^NoOp*
T0*
_output_shapes
: �1
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_100/DisableCopyOnRead^Read_100/ReadVariableOp^Read_101/DisableCopyOnRead^Read_101/ReadVariableOp^Read_102/DisableCopyOnRead^Read_102/ReadVariableOp^Read_103/DisableCopyOnRead^Read_103/ReadVariableOp^Read_104/DisableCopyOnRead^Read_104/ReadVariableOp^Read_105/DisableCopyOnRead^Read_105/ReadVariableOp^Read_106/DisableCopyOnRead^Read_106/ReadVariableOp^Read_107/DisableCopyOnRead^Read_107/ReadVariableOp^Read_108/DisableCopyOnRead^Read_108/ReadVariableOp^Read_109/DisableCopyOnRead^Read_109/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_110/DisableCopyOnRead^Read_110/ReadVariableOp^Read_111/DisableCopyOnRead^Read_111/ReadVariableOp^Read_112/DisableCopyOnRead^Read_112/ReadVariableOp^Read_113/DisableCopyOnRead^Read_113/ReadVariableOp^Read_114/DisableCopyOnRead^Read_114/ReadVariableOp^Read_115/DisableCopyOnRead^Read_115/ReadVariableOp^Read_116/DisableCopyOnRead^Read_116/ReadVariableOp^Read_117/DisableCopyOnRead^Read_117/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp^Read_96/DisableCopyOnRead^Read_96/ReadVariableOp^Read_97/DisableCopyOnRead^Read_97/ReadVariableOp^Read_98/DisableCopyOnRead^Read_98/ReadVariableOp^Read_99/DisableCopyOnRead^Read_99/ReadVariableOp*
_output_shapes
 "%
identity_237Identity_237:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp28
Read_100/DisableCopyOnReadRead_100/DisableCopyOnRead22
Read_100/ReadVariableOpRead_100/ReadVariableOp28
Read_101/DisableCopyOnReadRead_101/DisableCopyOnRead22
Read_101/ReadVariableOpRead_101/ReadVariableOp28
Read_102/DisableCopyOnReadRead_102/DisableCopyOnRead22
Read_102/ReadVariableOpRead_102/ReadVariableOp28
Read_103/DisableCopyOnReadRead_103/DisableCopyOnRead22
Read_103/ReadVariableOpRead_103/ReadVariableOp28
Read_104/DisableCopyOnReadRead_104/DisableCopyOnRead22
Read_104/ReadVariableOpRead_104/ReadVariableOp28
Read_105/DisableCopyOnReadRead_105/DisableCopyOnRead22
Read_105/ReadVariableOpRead_105/ReadVariableOp28
Read_106/DisableCopyOnReadRead_106/DisableCopyOnRead22
Read_106/ReadVariableOpRead_106/ReadVariableOp28
Read_107/DisableCopyOnReadRead_107/DisableCopyOnRead22
Read_107/ReadVariableOpRead_107/ReadVariableOp28
Read_108/DisableCopyOnReadRead_108/DisableCopyOnRead22
Read_108/ReadVariableOpRead_108/ReadVariableOp28
Read_109/DisableCopyOnReadRead_109/DisableCopyOnRead22
Read_109/ReadVariableOpRead_109/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp28
Read_110/DisableCopyOnReadRead_110/DisableCopyOnRead22
Read_110/ReadVariableOpRead_110/ReadVariableOp28
Read_111/DisableCopyOnReadRead_111/DisableCopyOnRead22
Read_111/ReadVariableOpRead_111/ReadVariableOp28
Read_112/DisableCopyOnReadRead_112/DisableCopyOnRead22
Read_112/ReadVariableOpRead_112/ReadVariableOp28
Read_113/DisableCopyOnReadRead_113/DisableCopyOnRead22
Read_113/ReadVariableOpRead_113/ReadVariableOp28
Read_114/DisableCopyOnReadRead_114/DisableCopyOnRead22
Read_114/ReadVariableOpRead_114/ReadVariableOp28
Read_115/DisableCopyOnReadRead_115/DisableCopyOnRead22
Read_115/ReadVariableOpRead_115/ReadVariableOp28
Read_116/DisableCopyOnReadRead_116/DisableCopyOnRead22
Read_116/ReadVariableOpRead_116/ReadVariableOp28
Read_117/DisableCopyOnReadRead_117/DisableCopyOnRead22
Read_117/ReadVariableOpRead_117/ReadVariableOp26
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
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp26
Read_96/DisableCopyOnReadRead_96/DisableCopyOnRead20
Read_96/ReadVariableOpRead_96/ReadVariableOp26
Read_97/DisableCopyOnReadRead_97/DisableCopyOnRead20
Read_97/ReadVariableOpRead_97/ReadVariableOp26
Read_98/DisableCopyOnReadRead_98/DisableCopyOnRead20
Read_98/ReadVariableOpRead_98/ReadVariableOp26
Read_99/DisableCopyOnReadRead_99/DisableCopyOnRead20
Read_99/ReadVariableOpRead_99/ReadVariableOp:?w;

_output_shapes
: 
!
_user_specified_name	Const_1:%v!

_user_specified_namecount:%u!

_user_specified_nametotal:'t#
!
_user_specified_name	count_1:'s#
!
_user_specified_name	total_1:4r0
.
_user_specified_nameAdam/v/dense_22/bias:4q0
.
_user_specified_nameAdam/m/dense_22/bias:6p2
0
_user_specified_nameAdam/v/dense_22/kernel:6o2
0
_user_specified_nameAdam/m/dense_22/kernel:Bn>
<
_user_specified_name$"Adam/v/layer_normalization_15/beta:Bm>
<
_user_specified_name$"Adam/m/layer_normalization_15/beta:Cl?
=
_user_specified_name%#Adam/v/layer_normalization_15/gamma:Ck?
=
_user_specified_name%#Adam/m/layer_normalization_15/gamma:4j0
.
_user_specified_nameAdam/v/dense_21/bias:4i0
.
_user_specified_nameAdam/m/dense_21/bias:6h2
0
_user_specified_nameAdam/v/dense_21/kernel:6g2
0
_user_specified_nameAdam/m/dense_21/kernel:Af=
;
_user_specified_name#!Adam/v/batch_normalization_1/beta:Ae=
;
_user_specified_name#!Adam/m/batch_normalization_1/beta:Bd>
<
_user_specified_name$"Adam/v/batch_normalization_1/gamma:Bc>
<
_user_specified_name$"Adam/m/batch_normalization_1/gamma:6b2
0
_user_specified_nameAdam/v/dense_20/kernel:6a2
0
_user_specified_nameAdam/m/dense_20/kernel:B`>
<
_user_specified_name$"Adam/v/layer_normalization_14/beta:B_>
<
_user_specified_name$"Adam/m/layer_normalization_14/beta:C^?
=
_user_specified_name%#Adam/v/layer_normalization_14/gamma:C]?
=
_user_specified_name%#Adam/m/layer_normalization_14/gamma:S\O
M
_user_specified_name53Adam/v/multi_head_attention_7/attention_output/bias:S[O
M
_user_specified_name53Adam/m/multi_head_attention_7/attention_output/bias:UZQ
O
_user_specified_name75Adam/v/multi_head_attention_7/attention_output/kernel:UYQ
O
_user_specified_name75Adam/m/multi_head_attention_7/attention_output/kernel:HXD
B
_user_specified_name*(Adam/v/multi_head_attention_7/value/bias:HWD
B
_user_specified_name*(Adam/m/multi_head_attention_7/value/bias:JVF
D
_user_specified_name,*Adam/v/multi_head_attention_7/value/kernel:JUF
D
_user_specified_name,*Adam/m/multi_head_attention_7/value/kernel:FTB
@
_user_specified_name(&Adam/v/multi_head_attention_7/key/bias:FSB
@
_user_specified_name(&Adam/m/multi_head_attention_7/key/bias:HRD
B
_user_specified_name*(Adam/v/multi_head_attention_7/key/kernel:HQD
B
_user_specified_name*(Adam/m/multi_head_attention_7/key/kernel:HPD
B
_user_specified_name*(Adam/v/multi_head_attention_7/query/bias:HOD
B
_user_specified_name*(Adam/m/multi_head_attention_7/query/bias:JNF
D
_user_specified_name,*Adam/v/multi_head_attention_7/query/kernel:JMF
D
_user_specified_name,*Adam/m/multi_head_attention_7/query/kernel:BL>
<
_user_specified_name$"Adam/v/layer_normalization_13/beta:BK>
<
_user_specified_name$"Adam/m/layer_normalization_13/beta:CJ?
=
_user_specified_name%#Adam/v/layer_normalization_13/gamma:CI?
=
_user_specified_name%#Adam/m/layer_normalization_13/gamma:4H0
.
_user_specified_nameAdam/v/dense_19/bias:4G0
.
_user_specified_nameAdam/m/dense_19/bias:6F2
0
_user_specified_nameAdam/v/dense_19/kernel:6E2
0
_user_specified_nameAdam/m/dense_19/kernel:?D;
9
_user_specified_name!Adam/v/batch_normalization/beta:?C;
9
_user_specified_name!Adam/m/batch_normalization/beta:@B<
:
_user_specified_name" Adam/v/batch_normalization/gamma:@A<
:
_user_specified_name" Adam/m/batch_normalization/gamma:6@2
0
_user_specified_nameAdam/v/dense_18/kernel:6?2
0
_user_specified_nameAdam/m/dense_18/kernel:B>>
<
_user_specified_name$"Adam/v/layer_normalization_12/beta:B=>
<
_user_specified_name$"Adam/m/layer_normalization_12/beta:C<?
=
_user_specified_name%#Adam/v/layer_normalization_12/gamma:C;?
=
_user_specified_name%#Adam/m/layer_normalization_12/gamma:S:O
M
_user_specified_name53Adam/v/multi_head_attention_6/attention_output/bias:S9O
M
_user_specified_name53Adam/m/multi_head_attention_6/attention_output/bias:U8Q
O
_user_specified_name75Adam/v/multi_head_attention_6/attention_output/kernel:U7Q
O
_user_specified_name75Adam/m/multi_head_attention_6/attention_output/kernel:H6D
B
_user_specified_name*(Adam/v/multi_head_attention_6/value/bias:H5D
B
_user_specified_name*(Adam/m/multi_head_attention_6/value/bias:J4F
D
_user_specified_name,*Adam/v/multi_head_attention_6/value/kernel:J3F
D
_user_specified_name,*Adam/m/multi_head_attention_6/value/kernel:F2B
@
_user_specified_name(&Adam/v/multi_head_attention_6/key/bias:F1B
@
_user_specified_name(&Adam/m/multi_head_attention_6/key/bias:H0D
B
_user_specified_name*(Adam/v/multi_head_attention_6/key/kernel:H/D
B
_user_specified_name*(Adam/m/multi_head_attention_6/key/kernel:H.D
B
_user_specified_name*(Adam/v/multi_head_attention_6/query/bias:H-D
B
_user_specified_name*(Adam/m/multi_head_attention_6/query/bias:J,F
D
_user_specified_name,*Adam/v/multi_head_attention_6/query/kernel:J+F
D
_user_specified_name,*Adam/m/multi_head_attention_6/query/kernel:-*)
'
_user_specified_namelearning_rate:))%
#
_user_specified_name	iteration:L(H
F
_user_specified_name.,multi_head_attention_7/attention_output/bias:N'J
H
_user_specified_name0.multi_head_attention_7/attention_output/kernel:A&=
;
_user_specified_name#!multi_head_attention_7/value/bias:C%?
=
_user_specified_name%#multi_head_attention_7/value/kernel:?$;
9
_user_specified_name!multi_head_attention_7/key/bias:A#=
;
_user_specified_name#!multi_head_attention_7/key/kernel:A"=
;
_user_specified_name#!multi_head_attention_7/query/bias:C!?
=
_user_specified_name%#multi_head_attention_7/query/kernel:L H
F
_user_specified_name.,multi_head_attention_6/attention_output/bias:NJ
H
_user_specified_name0.multi_head_attention_6/attention_output/kernel:A=
;
_user_specified_name#!multi_head_attention_6/value/bias:C?
=
_user_specified_name%#multi_head_attention_6/value/kernel:?;
9
_user_specified_name!multi_head_attention_6/key/bias:A=
;
_user_specified_name#!multi_head_attention_6/key/kernel:A=
;
_user_specified_name#!multi_head_attention_6/query/bias:C?
=
_user_specified_name%#multi_head_attention_6/query/kernel:-)
'
_user_specified_namedense_22/bias:/+
)
_user_specified_namedense_22/kernel:;7
5
_user_specified_namelayer_normalization_15/beta:<8
6
_user_specified_namelayer_normalization_15/gamma:-)
'
_user_specified_namedense_21/bias:/+
)
_user_specified_namedense_21/kernel:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean::6
4
_user_specified_namebatch_normalization_1/beta:;7
5
_user_specified_namebatch_normalization_1/gamma:/+
)
_user_specified_namedense_20/kernel:;7
5
_user_specified_namelayer_normalization_14/beta:<8
6
_user_specified_namelayer_normalization_14/gamma:;7
5
_user_specified_namelayer_normalization_13/beta:<
8
6
_user_specified_namelayer_normalization_13/gamma:-	)
'
_user_specified_namedense_19/bias:/+
)
_user_specified_namedense_19/kernel:C?
=
_user_specified_name%#batch_normalization/moving_variance:?;
9
_user_specified_name!batch_normalization/moving_mean:84
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:/+
)
_user_specified_namedense_18/kernel:;7
5
_user_specified_namelayer_normalization_12/beta:<8
6
_user_specified_namelayer_normalization_12/gamma:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
6__inference_batch_normalization_1_layer_call_fn_516938

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_515082|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name516934:&"
 
_user_specified_name516932:&"
 
_user_specified_name516930:&"
 
_user_specified_name516928:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
7__inference_layer_normalization_13_layer_call_fn_516692

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_layer_normalization_13_layer_call_and_return_conditional_losses_515363s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name516688:&"
 
_user_specified_name516686:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�,
�
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_515647	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������>*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *:͓>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������>�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������>>*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������>>q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������>>�
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������>*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������>*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:���������>�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������>:���������>: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
+
_output_shapes
:���������>

_user_specified_namevalue:R N
+
_output_shapes
:���������>

_user_specified_namequery
�
�
D__inference_dense_20_layer_call_and_return_conditional_losses_515488

inputs3
!tensordot_readvariableop_resource:
identity��Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������>�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������>e
IdentityIdentityTensordot:output:0^NoOp*
T0*+
_output_shapes
:���������>=
NoOpNoOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������>: 24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_516627

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������>Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������>*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������>T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:���������>e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
D__inference_dense_19_layer_call_and_return_conditional_losses_515329

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������>�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������>r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������>V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�&
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_516575

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_515688

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������>_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������>"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������>:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_515002

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
l
B__inference_add_13_layer_call_and_return_conditional_losses_515340

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������>S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������>:���������>:SO
+
_output_shapes
:���������>
 
_user_specified_nameinputs:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_517111

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������>�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������>l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������>a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������>~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������>f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������>\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
7__inference_layer_normalization_12_layer_call_fn_516459

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_layer_normalization_12_layer_call_and_return_conditional_losses_515236s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name516455:&"
 
_user_specified_name516453:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�
�
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_515583

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������>�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������>l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������>*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������>a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������>~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������>g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������>v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������>f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:���������>\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������>: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�'
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_516972

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
n
B__inference_add_14_layer_call_and_return_conditional_losses_516847
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������>S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������>:���������>:UQ
+
_output_shapes
:���������>
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������>
"
_user_specified_name
inputs_0
�
�
7__inference_multi_head_attention_6_layer_call_fn_516361	
query	
value
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_515647s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������>:���������>: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&	"
 
_user_specified_name516357:&"
 
_user_specified_name516355:&"
 
_user_specified_name516353:&"
 
_user_specified_name516351:&"
 
_user_specified_name516349:&"
 
_user_specified_name516347:&"
 
_user_specified_name516345:&"
 
_user_specified_name516343:RN
+
_output_shapes
:���������>

_user_specified_namevalue:R N
+
_output_shapes
:���������>

_user_specified_namequery
�	
�
4__inference_batch_normalization_layer_call_fn_516541

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_515002|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name516537:&"
 
_user_specified_name516535:&"
 
_user_specified_name516533:&"
 
_user_specified_name516531:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
n
B__inference_add_12_layer_call_and_return_conditional_losses_516450
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������>S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������>:���������>:UQ
+
_output_shapes
:���������>
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������>
"
_user_specified_name
inputs_0
�
�
7__inference_multi_head_attention_6_layer_call_fn_516339	
query	
value
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_515190s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������><
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������>:���������>: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&	"
 
_user_specified_name516335:&"
 
_user_specified_name516333:&"
 
_user_specified_name516331:&"
 
_user_specified_name516329:&"
 
_user_specified_name516327:&"
 
_user_specified_name516325:&"
 
_user_specified_name516323:&"
 
_user_specified_name516321:RN
+
_output_shapes
:���������>

_user_specified_namevalue:R N
+
_output_shapes
:���������>

_user_specified_namequery
�
z
Q__inference_positional_encoding_6_layer_call_and_return_conditional_losses_515145

inputs
unknown
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
ConstConst*
_output_shapes
: *
dtype0*
value	B : I
Const_1Const*
_output_shapes
: *
dtype0*
value	B :Y
strided_slice_1/stack/0Const*
_output_shapes
: *
dtype0*
value	B : Y
strided_slice_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : �
strided_slice_1/stackPack strided_slice_1/stack/0:output:0Const:output:0 strided_slice_1/stack/2:output:0*
N*
T0*
_output_shapes
:[
strided_slice_1/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : [
strided_slice_1/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : �
strided_slice_1/stack_1Pack"strided_slice_1/stack_1/0:output:0strided_slice:output:0"strided_slice_1/stack_1/2:output:0*
N*
T0*
_output_shapes
:[
strided_slice_1/stack_2/0Const*
_output_shapes
: *
dtype0*
value	B :[
strided_slice_1/stack_2/2Const*
_output_shapes
: *
dtype0*
value	B :�
strided_slice_1/stack_2Pack"strided_slice_1/stack_2/0:output:0Const_1:output:0"strided_slice_1/stack_2/2:output:0*
N*
T0*
_output_shapes
:�
strided_slice_1StridedSliceunknownstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*"
_output_shapes
:>*

begin_mask*
end_maskd
addAddV2inputsstrided_slice_1:output:0*
T0*+
_output_shapes
:���������>S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������>:>:JF
"
_output_shapes
:>
 
_user_specified_name515129:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs
�&
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_514982

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
��
�
C__inference_model_6_layer_call_and_return_conditional_losses_515796
input_7 
positional_encoding_6_5156103
multi_head_attention_6_515648:/
multi_head_attention_6_515650:3
multi_head_attention_6_515652:/
multi_head_attention_6_515654:3
multi_head_attention_6_515656:/
multi_head_attention_6_515658:3
multi_head_attention_6_515660:+
multi_head_attention_6_515662:+
layer_normalization_12_515666:+
layer_normalization_12_515668:!
dense_18_515671:(
batch_normalization_515674:(
batch_normalization_515676:(
batch_normalization_515678:(
batch_normalization_515680:!
dense_19_515690:
dense_19_515692:+
layer_normalization_13_515696:+
layer_normalization_13_515698:3
multi_head_attention_7_515736:/
multi_head_attention_7_515738:3
multi_head_attention_7_515740:/
multi_head_attention_7_515742:3
multi_head_attention_7_515744:/
multi_head_attention_7_515746:3
multi_head_attention_7_515748:+
multi_head_attention_7_515750:+
layer_normalization_14_515754:+
layer_normalization_14_515756:!
dense_20_515759:*
batch_normalization_1_515762:*
batch_normalization_1_515764:*
batch_normalization_1_515766:*
batch_normalization_1_515768:!
dense_21_515778:
dense_21_515780:+
layer_normalization_15_515784:+
layer_normalization_15_515786:!
dense_22_515790:
dense_22_515792:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall�.layer_normalization_12/StatefulPartitionedCall�.layer_normalization_13/StatefulPartitionedCall�.layer_normalization_14/StatefulPartitionedCall�.layer_normalization_15/StatefulPartitionedCall�.multi_head_attention_6/StatefulPartitionedCall�.multi_head_attention_7/StatefulPartitionedCall�
%positional_encoding_6/PartitionedCallPartitionedCallinput_7positional_encoding_6_515610*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_positional_encoding_6_layer_call_and_return_conditional_losses_515145�
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall.positional_encoding_6/PartitionedCall:output:0.positional_encoding_6/PartitionedCall:output:0multi_head_attention_6_515648multi_head_attention_6_515650multi_head_attention_6_515652multi_head_attention_6_515654multi_head_attention_6_515656multi_head_attention_6_515658multi_head_attention_6_515660multi_head_attention_6_515662*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_515647�
add_12/PartitionedCallPartitionedCall.positional_encoding_6/PartitionedCall:output:07multi_head_attention_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_add_12_layer_call_and_return_conditional_losses_515213�
.layer_normalization_12/StatefulPartitionedCallStatefulPartitionedCalladd_12/PartitionedCall:output:0layer_normalization_12_515666layer_normalization_12_515668*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_layer_normalization_12_layer_call_and_return_conditional_losses_515236�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:0dense_18_515671*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_515268�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0batch_normalization_515674batch_normalization_515676batch_normalization_515678batch_normalization_515680*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_515002�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_515285�
dropout/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_515688�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_19_515690dense_19_515692*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_515329�
add_13/PartitionedCallPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:0)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_add_13_layer_call_and_return_conditional_losses_515340�
.layer_normalization_13/StatefulPartitionedCallStatefulPartitionedCalladd_13/PartitionedCall:output:0layer_normalization_13_515696layer_normalization_13_515698*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_layer_normalization_13_layer_call_and_return_conditional_losses_515363�
.multi_head_attention_7/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_13/StatefulPartitionedCall:output:07layer_normalization_13/StatefulPartitionedCall:output:0multi_head_attention_7_515736multi_head_attention_7_515738multi_head_attention_7_515740multi_head_attention_7_515742multi_head_attention_7_515744multi_head_attention_7_515746multi_head_attention_7_515748multi_head_attention_7_515750*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_515735�
add_14/PartitionedCallPartitionedCall7layer_normalization_13/StatefulPartitionedCall:output:07multi_head_attention_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_add_14_layer_call_and_return_conditional_losses_515433�
.layer_normalization_14/StatefulPartitionedCallStatefulPartitionedCalladd_14/PartitionedCall:output:0layer_normalization_14_515754layer_normalization_14_515756*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_layer_normalization_14_layer_call_and_return_conditional_losses_515456�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_14/StatefulPartitionedCall:output:0dense_20_515759*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_515488�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_1_515762batch_normalization_1_515764batch_normalization_1_515766batch_normalization_1_515768*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_515082�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_515505�
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_515776�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_21_515778dense_21_515780*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_515549�
add_15/PartitionedCallPartitionedCall7layer_normalization_14/StatefulPartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_add_15_layer_call_and_return_conditional_losses_515560�
.layer_normalization_15/StatefulPartitionedCallStatefulPartitionedCalladd_15/PartitionedCall:output:0layer_normalization_15_515784layer_normalization_15_515786*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_515583�
*global_average_pooling1d_6/PartitionedCallPartitionedCall7layer_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *_
fZRX
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_515114�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_6/PartitionedCall:output:0dense_22_515790dense_22_515792*
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
D__inference_dense_22_layer_call_and_return_conditional_losses_515600x
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall/^layer_normalization_12/StatefulPartitionedCall/^layer_normalization_13/StatefulPartitionedCall/^layer_normalization_14/StatefulPartitionedCall/^layer_normalization_15/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall/^multi_head_attention_7/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesw
u:���������>:>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2`
.layer_normalization_12/StatefulPartitionedCall.layer_normalization_12/StatefulPartitionedCall2`
.layer_normalization_13/StatefulPartitionedCall.layer_normalization_13/StatefulPartitionedCall2`
.layer_normalization_14/StatefulPartitionedCall.layer_normalization_14/StatefulPartitionedCall2`
.layer_normalization_15/StatefulPartitionedCall.layer_normalization_15/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2`
.multi_head_attention_7/StatefulPartitionedCall.multi_head_attention_7/StatefulPartitionedCall:&)"
 
_user_specified_name515792:&("
 
_user_specified_name515790:&'"
 
_user_specified_name515786:&&"
 
_user_specified_name515784:&%"
 
_user_specified_name515780:&$"
 
_user_specified_name515778:&#"
 
_user_specified_name515768:&""
 
_user_specified_name515766:&!"
 
_user_specified_name515764:& "
 
_user_specified_name515762:&"
 
_user_specified_name515759:&"
 
_user_specified_name515756:&"
 
_user_specified_name515754:&"
 
_user_specified_name515750:&"
 
_user_specified_name515748:&"
 
_user_specified_name515746:&"
 
_user_specified_name515744:&"
 
_user_specified_name515742:&"
 
_user_specified_name515740:&"
 
_user_specified_name515738:&"
 
_user_specified_name515736:&"
 
_user_specified_name515698:&"
 
_user_specified_name515696:&"
 
_user_specified_name515692:&"
 
_user_specified_name515690:&"
 
_user_specified_name515680:&"
 
_user_specified_name515678:&"
 
_user_specified_name515676:&"
 
_user_specified_name515674:&"
 
_user_specified_name515671:&"
 
_user_specified_name515668:&
"
 
_user_specified_name515666:&	"
 
_user_specified_name515662:&"
 
_user_specified_name515660:&"
 
_user_specified_name515658:&"
 
_user_specified_name515656:&"
 
_user_specified_name515654:&"
 
_user_specified_name515652:&"
 
_user_specified_name515650:&"
 
_user_specified_name515648:JF
"
_output_shapes
:>
 
_user_specified_name515610:T P
+
_output_shapes
:���������>
!
_user_specified_name	input_7
�
�
D__inference_dense_20_layer_call_and_return_conditional_losses_516912

inputs3
!tensordot_readvariableop_resource:
identity��Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������>�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������>e
IdentityIdentityTensordot:output:0^NoOp*
T0*+
_output_shapes
:���������>=
NoOpNoOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������>: 24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������>
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_74
serving_default_input_7:0���������><
dense_220
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer-17
layer-18
layer_with_weights-10
layer-19
layer-20
layer_with_weights-11
layer-21
layer-22
layer_with_weights-12
layer-23
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
 	optimizer
!
signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
._query_dense
/
_key_dense
0_value_dense
1_softmax
2_dropout_layer
3_output_dense"
_tf_keras_layer
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@axis
	Agamma
Bbeta"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel"
_tf_keras_layer
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_random_generator"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias"
_tf_keras_layer
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses
vaxis
	wgamma
xbeta"
_tf_keras_layer
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�0
�1
�2
�3
�4
�5
�6
�7
A8
B9
I10
Q11
R12
S13
T14
h15
i16
w17
x18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
A8
B9
I10
Q11
R12
h13
i14
w15
x16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_model_6_layer_call_fn_515883
(__inference_model_6_layer_call_fn_515970�
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
�
�trace_0
�trace_12�
C__inference_model_6_layer_call_and_return_conditional_losses_515607
C__inference_model_6_layer_call_and_return_conditional_losses_515796�
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
�
�	capture_0B�
!__inference__wrapped_model_514948input_7"�
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
 z�	capture_0
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
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
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_positional_encoding_6_layer_call_fn_516293�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Q__inference_positional_encoding_6_layer_call_and_return_conditional_losses_516317�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_multi_head_attention_6_layer_call_fn_516339
7__inference_multi_head_attention_6_layer_call_fn_516361�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_516403
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_516438�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_add_12_layer_call_fn_516444�
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
 z�trace_0
�
�trace_02�
B__inference_add_12_layer_call_and_return_conditional_losses_516450�
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
 z�trace_0
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
7__inference_layer_normalization_12_layer_call_fn_516459�
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
 z�trace_0
�
�trace_02�
R__inference_layer_normalization_12_layer_call_and_return_conditional_losses_516481�
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
 z�trace_0
 "
trackable_list_wrapper
*:(2layer_normalization_12/gamma
):'2layer_normalization_12/beta
'
I0"
trackable_list_wrapper
'
I0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_18_layer_call_fn_516488�
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
 z�trace_0
�
�trace_02�
D__inference_dense_18_layer_call_and_return_conditional_losses_516515�
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
 z�trace_0
!:2dense_18/kernel
<
Q0
R1
S2
T3"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_batch_normalization_layer_call_fn_516528
4__inference_batch_normalization_layer_call_fn_516541�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_516575
O__inference_batch_normalization_layer_call_and_return_conditional_losses_516595�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_activation_layer_call_fn_516600�
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
 z�trace_0
�
�trace_02�
F__inference_activation_layer_call_and_return_conditional_losses_516605�
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
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_layer_call_fn_516610
(__inference_dropout_layer_call_fn_516615�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_layer_call_and_return_conditional_losses_516627
C__inference_dropout_layer_call_and_return_conditional_losses_516632�
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
 z�trace_0z�trace_1
"
_generic_user_object
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_19_layer_call_fn_516641�
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
 z�trace_0
�
�trace_02�
D__inference_dense_19_layer_call_and_return_conditional_losses_516671�
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
 z�trace_0
!:2dense_19/kernel
:2dense_19/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_add_13_layer_call_fn_516677�
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
 z�trace_0
�
�trace_02�
B__inference_add_13_layer_call_and_return_conditional_losses_516683�
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
 z�trace_0
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
7__inference_layer_normalization_13_layer_call_fn_516692�
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
 z�trace_0
�
�trace_02�
R__inference_layer_normalization_13_layer_call_and_return_conditional_losses_516714�
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
 z�trace_0
 "
trackable_list_wrapper
*:(2layer_normalization_13/gamma
):'2layer_normalization_13/beta
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_multi_head_attention_7_layer_call_fn_516736
7__inference_multi_head_attention_7_layer_call_fn_516758�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_516800
R__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_516835�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_add_14_layer_call_fn_516841�
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
 z�trace_0
�
�trace_02�
B__inference_add_14_layer_call_and_return_conditional_losses_516847�
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
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
7__inference_layer_normalization_14_layer_call_fn_516856�
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
 z�trace_0
�
�trace_02�
R__inference_layer_normalization_14_layer_call_and_return_conditional_losses_516878�
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
 z�trace_0
 "
trackable_list_wrapper
*:(2layer_normalization_14/gamma
):'2layer_normalization_14/beta
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_20_layer_call_fn_516885�
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
 z�trace_0
�
�trace_02�
D__inference_dense_20_layer_call_and_return_conditional_losses_516912�
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
 z�trace_0
!:2dense_20/kernel
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_1_layer_call_fn_516925
6__inference_batch_normalization_1_layer_call_fn_516938�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_516972
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_516992�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_activation_1_layer_call_fn_516997�
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
 z�trace_0
�
�trace_02�
H__inference_activation_1_layer_call_and_return_conditional_losses_517002�
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
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_1_layer_call_fn_517007
*__inference_dropout_1_layer_call_fn_517012�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_1_layer_call_and_return_conditional_losses_517024
E__inference_dropout_1_layer_call_and_return_conditional_losses_517029�
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
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_21_layer_call_fn_517038�
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
 z�trace_0
�
�trace_02�
D__inference_dense_21_layer_call_and_return_conditional_losses_517068�
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
 z�trace_0
!:2dense_21/kernel
:2dense_21/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_add_15_layer_call_fn_517074�
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
 z�trace_0
�
�trace_02�
B__inference_add_15_layer_call_and_return_conditional_losses_517080�
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
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
7__inference_layer_normalization_15_layer_call_fn_517089�
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
 z�trace_0
�
�trace_02�
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_517111�
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
 z�trace_0
 "
trackable_list_wrapper
*:(2layer_normalization_15/gamma
):'2layer_normalization_15/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
;__inference_global_average_pooling1d_6_layer_call_fn_517116�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_517122�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_22_layer_call_fn_517131�
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
 z�trace_0
�
�trace_02�
D__inference_dense_22_layer_call_and_return_conditional_losses_517142�
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
 z�trace_0
!:2dense_22/kernel
:2dense_22/bias
9:72#multi_head_attention_6/query/kernel
3:12!multi_head_attention_6/query/bias
7:52!multi_head_attention_6/key/kernel
1:/2multi_head_attention_6/key/bias
9:72#multi_head_attention_6/value/kernel
3:12!multi_head_attention_6/value/bias
D:B2.multi_head_attention_6/attention_output/kernel
::82,multi_head_attention_6/attention_output/bias
9:72#multi_head_attention_7/query/kernel
3:12!multi_head_attention_7/query/bias
7:52!multi_head_attention_7/key/kernel
1:/2multi_head_attention_7/key/bias
9:72#multi_head_attention_7/value/kernel
3:12!multi_head_attention_7/value/bias
D:B2.multi_head_attention_7/attention_output/kernel
::82,multi_head_attention_7/attention_output/bias
>
S0
T1
�2
�3"
trackable_list_wrapper
�
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
10
11
12
13
14
15
16
17
18
19
20
21
22
23"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
�	capture_0B�
(__inference_model_6_layer_call_fn_515883input_7"�
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
 z�	capture_0
�
�	capture_0B�
(__inference_model_6_layer_call_fn_515970input_7"�
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
 z�	capture_0
�
�	capture_0B�
C__inference_model_6_layer_call_and_return_conditional_losses_515607input_7"�
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
 z�	capture_0
�
�	capture_0B�
C__inference_model_6_layer_call_and_return_conditional_losses_515796input_7"�
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
 z�	capture_0
J
Constjtf.TrackableConstant
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
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
�
�	capture_0B�
$__inference_signature_wrapper_516286input_7"�
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
 z�	capture_0
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
�
�	capture_0B�
6__inference_positional_encoding_6_layer_call_fn_516293inputs"�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0
�
�	capture_0B�
Q__inference_positional_encoding_6_layer_call_and_return_conditional_losses_516317inputs"�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0
 "
trackable_list_wrapper
J
.0
/1
02
13
24
35"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_multi_head_attention_6_layer_call_fn_516339queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_multi_head_attention_6_layer_call_fn_516361queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_516403queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_516438queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_add_12_layer_call_fn_516444inputs_0inputs_1"�
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
B__inference_add_12_layer_call_and_return_conditional_losses_516450inputs_0inputs_1"�
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
7__inference_layer_normalization_12_layer_call_fn_516459inputs"�
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
R__inference_layer_normalization_12_layer_call_and_return_conditional_losses_516481inputs"�
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
)__inference_dense_18_layer_call_fn_516488inputs"�
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
D__inference_dense_18_layer_call_and_return_conditional_losses_516515inputs"�
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
S0
T1"
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
4__inference_batch_normalization_layer_call_fn_516528inputs"�
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
4__inference_batch_normalization_layer_call_fn_516541inputs"�
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_516575inputs"�
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_516595inputs"�
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
+__inference_activation_layer_call_fn_516600inputs"�
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
F__inference_activation_layer_call_and_return_conditional_losses_516605inputs"�
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
(__inference_dropout_layer_call_fn_516610inputs"�
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
(__inference_dropout_layer_call_fn_516615inputs"�
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
C__inference_dropout_layer_call_and_return_conditional_losses_516627inputs"�
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
C__inference_dropout_layer_call_and_return_conditional_losses_516632inputs"�
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_19_layer_call_fn_516641inputs"�
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
D__inference_dense_19_layer_call_and_return_conditional_losses_516671inputs"�
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
'__inference_add_13_layer_call_fn_516677inputs_0inputs_1"�
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
B__inference_add_13_layer_call_and_return_conditional_losses_516683inputs_0inputs_1"�
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
7__inference_layer_normalization_13_layer_call_fn_516692inputs"�
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
R__inference_layer_normalization_13_layer_call_and_return_conditional_losses_516714inputs"�
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
O
0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_multi_head_attention_7_layer_call_fn_516736queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_multi_head_attention_7_layer_call_fn_516758queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_516800queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_516835queryvalue"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_add_14_layer_call_fn_516841inputs_0inputs_1"�
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
B__inference_add_14_layer_call_and_return_conditional_losses_516847inputs_0inputs_1"�
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
7__inference_layer_normalization_14_layer_call_fn_516856inputs"�
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
R__inference_layer_normalization_14_layer_call_and_return_conditional_losses_516878inputs"�
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
)__inference_dense_20_layer_call_fn_516885inputs"�
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
D__inference_dense_20_layer_call_and_return_conditional_losses_516912inputs"�
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
0
�0
�1"
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
6__inference_batch_normalization_1_layer_call_fn_516925inputs"�
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
6__inference_batch_normalization_1_layer_call_fn_516938inputs"�
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_516972inputs"�
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_516992inputs"�
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
-__inference_activation_1_layer_call_fn_516997inputs"�
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
H__inference_activation_1_layer_call_and_return_conditional_losses_517002inputs"�
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
*__inference_dropout_1_layer_call_fn_517007inputs"�
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
*__inference_dropout_1_layer_call_fn_517012inputs"�
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
E__inference_dropout_1_layer_call_and_return_conditional_losses_517024inputs"�
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
E__inference_dropout_1_layer_call_and_return_conditional_losses_517029inputs"�
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_21_layer_call_fn_517038inputs"�
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
D__inference_dense_21_layer_call_and_return_conditional_losses_517068inputs"�
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
'__inference_add_15_layer_call_fn_517074inputs_0inputs_1"�
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
B__inference_add_15_layer_call_and_return_conditional_losses_517080inputs_0inputs_1"�
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
7__inference_layer_normalization_15_layer_call_fn_517089inputs"�
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
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_517111inputs"�
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
;__inference_global_average_pooling1d_6_layer_call_fn_517116inputs"�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_517122inputs"�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�
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
)__inference_dense_22_layer_call_fn_517131inputs"�
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
D__inference_dense_22_layer_call_and_return_conditional_losses_517142inputs"�
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
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
>:<2*Adam/m/multi_head_attention_6/query/kernel
>:<2*Adam/v/multi_head_attention_6/query/kernel
8:62(Adam/m/multi_head_attention_6/query/bias
8:62(Adam/v/multi_head_attention_6/query/bias
<::2(Adam/m/multi_head_attention_6/key/kernel
<::2(Adam/v/multi_head_attention_6/key/kernel
6:42&Adam/m/multi_head_attention_6/key/bias
6:42&Adam/v/multi_head_attention_6/key/bias
>:<2*Adam/m/multi_head_attention_6/value/kernel
>:<2*Adam/v/multi_head_attention_6/value/kernel
8:62(Adam/m/multi_head_attention_6/value/bias
8:62(Adam/v/multi_head_attention_6/value/bias
I:G25Adam/m/multi_head_attention_6/attention_output/kernel
I:G25Adam/v/multi_head_attention_6/attention_output/kernel
?:=23Adam/m/multi_head_attention_6/attention_output/bias
?:=23Adam/v/multi_head_attention_6/attention_output/bias
/:-2#Adam/m/layer_normalization_12/gamma
/:-2#Adam/v/layer_normalization_12/gamma
.:,2"Adam/m/layer_normalization_12/beta
.:,2"Adam/v/layer_normalization_12/beta
&:$2Adam/m/dense_18/kernel
&:$2Adam/v/dense_18/kernel
,:*2 Adam/m/batch_normalization/gamma
,:*2 Adam/v/batch_normalization/gamma
+:)2Adam/m/batch_normalization/beta
+:)2Adam/v/batch_normalization/beta
&:$2Adam/m/dense_19/kernel
&:$2Adam/v/dense_19/kernel
 :2Adam/m/dense_19/bias
 :2Adam/v/dense_19/bias
/:-2#Adam/m/layer_normalization_13/gamma
/:-2#Adam/v/layer_normalization_13/gamma
.:,2"Adam/m/layer_normalization_13/beta
.:,2"Adam/v/layer_normalization_13/beta
>:<2*Adam/m/multi_head_attention_7/query/kernel
>:<2*Adam/v/multi_head_attention_7/query/kernel
8:62(Adam/m/multi_head_attention_7/query/bias
8:62(Adam/v/multi_head_attention_7/query/bias
<::2(Adam/m/multi_head_attention_7/key/kernel
<::2(Adam/v/multi_head_attention_7/key/kernel
6:42&Adam/m/multi_head_attention_7/key/bias
6:42&Adam/v/multi_head_attention_7/key/bias
>:<2*Adam/m/multi_head_attention_7/value/kernel
>:<2*Adam/v/multi_head_attention_7/value/kernel
8:62(Adam/m/multi_head_attention_7/value/bias
8:62(Adam/v/multi_head_attention_7/value/bias
I:G25Adam/m/multi_head_attention_7/attention_output/kernel
I:G25Adam/v/multi_head_attention_7/attention_output/kernel
?:=23Adam/m/multi_head_attention_7/attention_output/bias
?:=23Adam/v/multi_head_attention_7/attention_output/bias
/:-2#Adam/m/layer_normalization_14/gamma
/:-2#Adam/v/layer_normalization_14/gamma
.:,2"Adam/m/layer_normalization_14/beta
.:,2"Adam/v/layer_normalization_14/beta
&:$2Adam/m/dense_20/kernel
&:$2Adam/v/dense_20/kernel
.:,2"Adam/m/batch_normalization_1/gamma
.:,2"Adam/v/batch_normalization_1/gamma
-:+2!Adam/m/batch_normalization_1/beta
-:+2!Adam/v/batch_normalization_1/beta
&:$2Adam/m/dense_21/kernel
&:$2Adam/v/dense_21/kernel
 :2Adam/m/dense_21/bias
 :2Adam/v/dense_21/bias
/:-2#Adam/m/layer_normalization_15/gamma
/:-2#Adam/v/layer_normalization_15/gamma
.:,2"Adam/m/layer_normalization_15/beta
.:,2"Adam/v/layer_normalization_15/beta
&:$2Adam/m/dense_22/kernel
&:$2Adam/v/dense_22/kernel
 :2Adam/m/dense_22/bias
 :2Adam/v/dense_22/bias
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
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
!__inference__wrapped_model_514948�G���������ABITQSRhiwx���������������������4�1
*�'
%�"
input_7���������>
� "3�0
.
dense_22"�
dense_22����������
H__inference_activation_1_layer_call_and_return_conditional_losses_517002g3�0
)�&
$�!
inputs���������>
� "0�-
&�#
tensor_0���������>
� �
-__inference_activation_1_layer_call_fn_516997\3�0
)�&
$�!
inputs���������>
� "%�"
unknown���������>�
F__inference_activation_layer_call_and_return_conditional_losses_516605g3�0
)�&
$�!
inputs���������>
� "0�-
&�#
tensor_0���������>
� �
+__inference_activation_layer_call_fn_516600\3�0
)�&
$�!
inputs���������>
� "%�"
unknown���������>�
B__inference_add_12_layer_call_and_return_conditional_losses_516450�b�_
X�U
S�P
&�#
inputs_0���������>
&�#
inputs_1���������>
� "0�-
&�#
tensor_0���������>
� �
'__inference_add_12_layer_call_fn_516444�b�_
X�U
S�P
&�#
inputs_0���������>
&�#
inputs_1���������>
� "%�"
unknown���������>�
B__inference_add_13_layer_call_and_return_conditional_losses_516683�b�_
X�U
S�P
&�#
inputs_0���������>
&�#
inputs_1���������>
� "0�-
&�#
tensor_0���������>
� �
'__inference_add_13_layer_call_fn_516677�b�_
X�U
S�P
&�#
inputs_0���������>
&�#
inputs_1���������>
� "%�"
unknown���������>�
B__inference_add_14_layer_call_and_return_conditional_losses_516847�b�_
X�U
S�P
&�#
inputs_0���������>
&�#
inputs_1���������>
� "0�-
&�#
tensor_0���������>
� �
'__inference_add_14_layer_call_fn_516841�b�_
X�U
S�P
&�#
inputs_0���������>
&�#
inputs_1���������>
� "%�"
unknown���������>�
B__inference_add_15_layer_call_and_return_conditional_losses_517080�b�_
X�U
S�P
&�#
inputs_0���������>
&�#
inputs_1���������>
� "0�-
&�#
tensor_0���������>
� �
'__inference_add_15_layer_call_fn_517074�b�_
X�U
S�P
&�#
inputs_0���������>
&�#
inputs_1���������>
� "%�"
unknown���������>�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_516972�����D�A
:�7
-�*
inputs������������������
p

 
� "9�6
/�,
tensor_0������������������
� �
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_516992�����D�A
:�7
-�*
inputs������������������
p 

 
� "9�6
/�,
tensor_0������������������
� �
6__inference_batch_normalization_1_layer_call_fn_516925�����D�A
:�7
-�*
inputs������������������
p

 
� ".�+
unknown�������������������
6__inference_batch_normalization_1_layer_call_fn_516938�����D�A
:�7
-�*
inputs������������������
p 

 
� ".�+
unknown�������������������
O__inference_batch_normalization_layer_call_and_return_conditional_losses_516575�STQRD�A
:�7
-�*
inputs������������������
p

 
� "9�6
/�,
tensor_0������������������
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_516595�TQSRD�A
:�7
-�*
inputs������������������
p 

 
� "9�6
/�,
tensor_0������������������
� �
4__inference_batch_normalization_layer_call_fn_516528|STQRD�A
:�7
-�*
inputs������������������
p

 
� ".�+
unknown�������������������
4__inference_batch_normalization_layer_call_fn_516541|TQSRD�A
:�7
-�*
inputs������������������
p 

 
� ".�+
unknown�������������������
D__inference_dense_18_layer_call_and_return_conditional_losses_516515jI3�0
)�&
$�!
inputs���������>
� "0�-
&�#
tensor_0���������>
� �
)__inference_dense_18_layer_call_fn_516488_I3�0
)�&
$�!
inputs���������>
� "%�"
unknown���������>�
D__inference_dense_19_layer_call_and_return_conditional_losses_516671khi3�0
)�&
$�!
inputs���������>
� "0�-
&�#
tensor_0���������>
� �
)__inference_dense_19_layer_call_fn_516641`hi3�0
)�&
$�!
inputs���������>
� "%�"
unknown���������>�
D__inference_dense_20_layer_call_and_return_conditional_losses_516912k�3�0
)�&
$�!
inputs���������>
� "0�-
&�#
tensor_0���������>
� �
)__inference_dense_20_layer_call_fn_516885`�3�0
)�&
$�!
inputs���������>
� "%�"
unknown���������>�
D__inference_dense_21_layer_call_and_return_conditional_losses_517068m��3�0
)�&
$�!
inputs���������>
� "0�-
&�#
tensor_0���������>
� �
)__inference_dense_21_layer_call_fn_517038b��3�0
)�&
$�!
inputs���������>
� "%�"
unknown���������>�
D__inference_dense_22_layer_call_and_return_conditional_losses_517142e��/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_22_layer_call_fn_517131Z��/�,
%�"
 �
inputs���������
� "!�
unknown����������
E__inference_dropout_1_layer_call_and_return_conditional_losses_517024k7�4
-�*
$�!
inputs���������>
p
� "0�-
&�#
tensor_0���������>
� �
E__inference_dropout_1_layer_call_and_return_conditional_losses_517029k7�4
-�*
$�!
inputs���������>
p 
� "0�-
&�#
tensor_0���������>
� �
*__inference_dropout_1_layer_call_fn_517007`7�4
-�*
$�!
inputs���������>
p
� "%�"
unknown���������>�
*__inference_dropout_1_layer_call_fn_517012`7�4
-�*
$�!
inputs���������>
p 
� "%�"
unknown���������>�
C__inference_dropout_layer_call_and_return_conditional_losses_516627k7�4
-�*
$�!
inputs���������>
p
� "0�-
&�#
tensor_0���������>
� �
C__inference_dropout_layer_call_and_return_conditional_losses_516632k7�4
-�*
$�!
inputs���������>
p 
� "0�-
&�#
tensor_0���������>
� �
(__inference_dropout_layer_call_fn_516610`7�4
-�*
$�!
inputs���������>
p
� "%�"
unknown���������>�
(__inference_dropout_layer_call_fn_516615`7�4
-�*
$�!
inputs���������>
p 
� "%�"
unknown���������>�
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_517122�I�F
?�<
6�3
inputs'���������������������������

 
� "5�2
+�(
tensor_0������������������
� �
;__inference_global_average_pooling1d_6_layer_call_fn_517116wI�F
?�<
6�3
inputs'���������������������������

 
� "*�'
unknown�������������������
R__inference_layer_normalization_12_layer_call_and_return_conditional_losses_516481kAB3�0
)�&
$�!
inputs���������>
� "0�-
&�#
tensor_0���������>
� �
7__inference_layer_normalization_12_layer_call_fn_516459`AB3�0
)�&
$�!
inputs���������>
� "%�"
unknown���������>�
R__inference_layer_normalization_13_layer_call_and_return_conditional_losses_516714kwx3�0
)�&
$�!
inputs���������>
� "0�-
&�#
tensor_0���������>
� �
7__inference_layer_normalization_13_layer_call_fn_516692`wx3�0
)�&
$�!
inputs���������>
� "%�"
unknown���������>�
R__inference_layer_normalization_14_layer_call_and_return_conditional_losses_516878m��3�0
)�&
$�!
inputs���������>
� "0�-
&�#
tensor_0���������>
� �
7__inference_layer_normalization_14_layer_call_fn_516856b��3�0
)�&
$�!
inputs���������>
� "%�"
unknown���������>�
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_517111m��3�0
)�&
$�!
inputs���������>
� "0�-
&�#
tensor_0���������>
� �
7__inference_layer_normalization_15_layer_call_fn_517089b��3�0
)�&
$�!
inputs���������>
� "%�"
unknown���������>�
C__inference_model_6_layer_call_and_return_conditional_losses_515607�G���������ABISTQRhiwx���������������������<�9
2�/
%�"
input_7���������>
p

 
� ",�)
"�
tensor_0���������
� �
C__inference_model_6_layer_call_and_return_conditional_losses_515796�G���������ABITQSRhiwx���������������������<�9
2�/
%�"
input_7���������>
p 

 
� ",�)
"�
tensor_0���������
� �
(__inference_model_6_layer_call_fn_515883�G���������ABISTQRhiwx���������������������<�9
2�/
%�"
input_7���������>
p

 
� "!�
unknown����������
(__inference_model_6_layer_call_fn_515970�G���������ABITQSRhiwx���������������������<�9
2�/
%�"
input_7���������>
p 

 
� "!�
unknown����������
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_516403���������k�h
a�^
#� 
query���������>
#� 
value���������>

 

 
p 
p
p 
� "0�-
&�#
tensor_0���������>
� �
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_516438���������k�h
a�^
#� 
query���������>
#� 
value���������>

 

 
p 
p 
p 
� "0�-
&�#
tensor_0���������>
� �
7__inference_multi_head_attention_6_layer_call_fn_516339���������k�h
a�^
#� 
query���������>
#� 
value���������>

 

 
p 
p
p 
� "%�"
unknown���������>�
7__inference_multi_head_attention_6_layer_call_fn_516361���������k�h
a�^
#� 
query���������>
#� 
value���������>

 

 
p 
p 
p 
� "%�"
unknown���������>�
R__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_516800���������k�h
a�^
#� 
query���������>
#� 
value���������>

 

 
p 
p
p 
� "0�-
&�#
tensor_0���������>
� �
R__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_516835���������k�h
a�^
#� 
query���������>
#� 
value���������>

 

 
p 
p 
p 
� "0�-
&�#
tensor_0���������>
� �
7__inference_multi_head_attention_7_layer_call_fn_516736���������k�h
a�^
#� 
query���������>
#� 
value���������>

 

 
p 
p
p 
� "%�"
unknown���������>�
7__inference_multi_head_attention_7_layer_call_fn_516758���������k�h
a�^
#� 
query���������>
#� 
value���������>

 

 
p 
p 
p 
� "%�"
unknown���������>�
Q__inference_positional_encoding_6_layer_call_and_return_conditional_losses_516317o�7�4
-�*
$�!
inputs���������>

 
� "0�-
&�#
tensor_0���������>
� �
6__inference_positional_encoding_6_layer_call_fn_516293d�7�4
-�*
$�!
inputs���������>

 
� "%�"
unknown���������>�
$__inference_signature_wrapper_516286�G���������ABITQSRhiwx���������������������?�<
� 
5�2
0
input_7%�"
input_7���������>"3�0
.
dense_22"�
dense_22���������