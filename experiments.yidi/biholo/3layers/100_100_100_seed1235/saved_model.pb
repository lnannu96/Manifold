ьс
бЃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878
l
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_name
Variable
e
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes

:d*
dtype0
p

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_name
Variable_1
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:dd*
dtype0
p

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_name
Variable_2
i
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes

:dd*
dtype0

NoOpNoOp
й
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B

biholomorphic

layer1

layer2

layer3
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
R

trainable_variables
	variables
regularization_losses
	keras_api
Y
w
trainable_variables
	variables
regularization_losses
	keras_api
Y
w
trainable_variables
	variables
regularization_losses
	keras_api
Y
w
trainable_variables
	variables
regularization_losses
	keras_api

0
1
2

0
1
2
 
­
layer_metrics

layers
metrics
trainable_variables
	variables
regularization_losses
 non_trainable_variables
!layer_regularization_losses
 
 
 
 
­
"layer_metrics
#metrics

$layers

trainable_variables
	variables
regularization_losses
%non_trainable_variables
&layer_regularization_losses
A?
VARIABLE_VALUEVariable#layer1/w/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
­
'layer_metrics
(metrics

)layers
trainable_variables
	variables
regularization_losses
*non_trainable_variables
+layer_regularization_losses
CA
VARIABLE_VALUE
Variable_1#layer2/w/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
­
,layer_metrics
-metrics

.layers
trainable_variables
	variables
regularization_losses
/non_trainable_variables
0layer_regularization_losses
CA
VARIABLE_VALUE
Variable_2#layer3/w/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
­
1layer_metrics
2metrics

3layers
trainable_variables
	variables
regularization_losses
4non_trainable_variables
5layer_regularization_losses
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
л
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Variable
Variable_1
Variable_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_19535399
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOpConst*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_save_19535534
Р
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable
Variable_1
Variable_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__traced_restore_19535553Ђю
з

&__inference_signature_wrapper_19535399
input_1
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_195352452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
ќ

C__inference_dense_layer_call_and_return_conditional_losses_19535465

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџ::X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
э
!__inference__traced_save_19535534
file_prefix'
#savev2_variable_read_readvariableop)
%savev2_variable_1_read_readvariableop)
%savev2_variable_2_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_9492afa08e674c2881efb8d517bcb154/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ђ
valueBB#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB#layer2/w/.ATTRIBUTES/VARIABLE_VALUEB#layer3/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
SaveV2/shape_and_slicesА
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*5
_input_shapes$
": :d:dd:dd: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d:$ 

_output_shapes

:dd:$ 

_output_shapes

:dd:

_output_shapes
: 
Хq
ј
#__inference__wrapped_model_19535245
input_19
5kahler_potential_dense_matmul_readvariableop_resource;
7kahler_potential_dense_1_matmul_readvariableop_resource;
7kahler_potential_dense_2_matmul_readvariableop_resource
identity
#kahler_potential/biholomorphic/ConjConjinput_1*'
_output_shapes
:џџџџџџџџџ2%
#kahler_potential/biholomorphic/Conjі
,kahler_potential/biholomorphic/einsum/EinsumEinsuminput_1,kahler_potential/biholomorphic/Conj:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equation
ai,aj->aij2.
,kahler_potential/biholomorphic/einsum/EinsumД
7kahler_potential/biholomorphic/MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 29
7kahler_potential/biholomorphic/MatrixBandPart/num_lowerН
7kahler_potential/biholomorphic/MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ29
7kahler_potential/biholomorphic/MatrixBandPart/num_upperс
-kahler_potential/biholomorphic/MatrixBandPartMatrixBandPart5kahler_potential/biholomorphic/einsum/Einsum:output:0@kahler_potential/biholomorphic/MatrixBandPart/num_lower:output:0@kahler_potential/biholomorphic/MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2/
-kahler_potential/biholomorphic/MatrixBandPart­
,kahler_potential/biholomorphic/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2.
,kahler_potential/biholomorphic/Reshape/shapeњ
&kahler_potential/biholomorphic/ReshapeReshape4kahler_potential/biholomorphic/MatrixBandPart:band:05kahler_potential/biholomorphic/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&kahler_potential/biholomorphic/ReshapeЌ
#kahler_potential/biholomorphic/RealReal/kahler_potential/biholomorphic/Reshape:output:0*'
_output_shapes
:џџџџџџџџџ2%
#kahler_potential/biholomorphic/RealЌ
#kahler_potential/biholomorphic/ImagImag/kahler_potential/biholomorphic/Reshape:output:0*'
_output_shapes
:џџџџџџџџџ2%
#kahler_potential/biholomorphic/Imag
*kahler_potential/biholomorphic/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*kahler_potential/biholomorphic/concat/axisІ
%kahler_potential/biholomorphic/concatConcatV2,kahler_potential/biholomorphic/Real:output:0,kahler_potential/biholomorphic/Imag:output:03kahler_potential/biholomorphic/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ22'
%kahler_potential/biholomorphic/concatЏ
-kahler_potential/biholomorphic/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2/
-kahler_potential/biholomorphic/transpose/permћ
(kahler_potential/biholomorphic/transpose	Transpose.kahler_potential/biholomorphic/concat:output:06kahler_potential/biholomorphic/transpose/perm:output:0*
T0*'
_output_shapes
:2џџџџџџџџџ2*
(kahler_potential/biholomorphic/transposeЏ
"kahler_potential/biholomorphic/AbsAbs,kahler_potential/biholomorphic/transpose:y:0*
T0*'
_output_shapes
:2џџџџџџџџџ2$
"kahler_potential/biholomorphic/AbsЎ
4kahler_potential/biholomorphic/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4kahler_potential/biholomorphic/Sum/reduction_indicesл
"kahler_potential/biholomorphic/SumSum&kahler_potential/biholomorphic/Abs:y:0=kahler_potential/biholomorphic/Sum/reduction_indices:output:0*
T0*
_output_shapes
:22$
"kahler_potential/biholomorphic/Sum
%kahler_potential/biholomorphic/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%kahler_potential/biholomorphic/Less/yд
#kahler_potential/biholomorphic/LessLess+kahler_potential/biholomorphic/Sum:output:0.kahler_potential/biholomorphic/Less/y:output:0*
T0*
_output_shapes
:22%
#kahler_potential/biholomorphic/LessЉ
)kahler_potential/biholomorphic/LogicalNot
LogicalNot'kahler_potential/biholomorphic/Less:z:0*
_output_shapes
:22+
)kahler_potential/biholomorphic/LogicalNotЏ
&kahler_potential/biholomorphic/SqueezeSqueeze-kahler_potential/biholomorphic/LogicalNot:y:0*
T0
*
_output_shapes
:22(
&kahler_potential/biholomorphic/SqueezeТ
1kahler_potential/biholomorphic/boolean_mask/ShapeShape,kahler_potential/biholomorphic/transpose:y:0*
T0*
_output_shapes
:23
1kahler_potential/biholomorphic/boolean_mask/ShapeЬ
?kahler_potential/biholomorphic/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?kahler_potential/biholomorphic/boolean_mask/strided_slice/stackа
Akahler_potential/biholomorphic/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Akahler_potential/biholomorphic/boolean_mask/strided_slice/stack_1а
Akahler_potential/biholomorphic/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Akahler_potential/biholomorphic/boolean_mask/strided_slice/stack_2ж
9kahler_potential/biholomorphic/boolean_mask/strided_sliceStridedSlice:kahler_potential/biholomorphic/boolean_mask/Shape:output:0Hkahler_potential/biholomorphic/boolean_mask/strided_slice/stack:output:0Jkahler_potential/biholomorphic/boolean_mask/strided_slice/stack_1:output:0Jkahler_potential/biholomorphic/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2;
9kahler_potential/biholomorphic/boolean_mask/strided_sliceв
Bkahler_potential/biholomorphic/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bkahler_potential/biholomorphic/boolean_mask/Prod/reduction_indices
0kahler_potential/biholomorphic/boolean_mask/ProdProdBkahler_potential/biholomorphic/boolean_mask/strided_slice:output:0Kkahler_potential/biholomorphic/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 22
0kahler_potential/biholomorphic/boolean_mask/ProdЦ
3kahler_potential/biholomorphic/boolean_mask/Shape_1Shape,kahler_potential/biholomorphic/transpose:y:0*
T0*
_output_shapes
:25
3kahler_potential/biholomorphic/boolean_mask/Shape_1а
Akahler_potential/biholomorphic/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Akahler_potential/biholomorphic/boolean_mask/strided_slice_1/stackд
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_1д
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_2ђ
;kahler_potential/biholomorphic/boolean_mask/strided_slice_1StridedSlice<kahler_potential/biholomorphic/boolean_mask/Shape_1:output:0Jkahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack:output:0Lkahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_1:output:0Lkahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2=
;kahler_potential/biholomorphic/boolean_mask/strided_slice_1Ц
3kahler_potential/biholomorphic/boolean_mask/Shape_2Shape,kahler_potential/biholomorphic/transpose:y:0*
T0*
_output_shapes
:25
3kahler_potential/biholomorphic/boolean_mask/Shape_2а
Akahler_potential/biholomorphic/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
Akahler_potential/biholomorphic/boolean_mask/strided_slice_2/stackд
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_1д
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_2ђ
;kahler_potential/biholomorphic/boolean_mask/strided_slice_2StridedSlice<kahler_potential/biholomorphic/boolean_mask/Shape_2:output:0Jkahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack:output:0Lkahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_1:output:0Lkahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2=
;kahler_potential/biholomorphic/boolean_mask/strided_slice_2ы
;kahler_potential/biholomorphic/boolean_mask/concat/values_1Pack9kahler_potential/biholomorphic/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2=
;kahler_potential/biholomorphic/boolean_mask/concat/values_1Д
7kahler_potential/biholomorphic/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7kahler_potential/biholomorphic/boolean_mask/concat/axisЖ
2kahler_potential/biholomorphic/boolean_mask/concatConcatV2Dkahler_potential/biholomorphic/boolean_mask/strided_slice_1:output:0Dkahler_potential/biholomorphic/boolean_mask/concat/values_1:output:0Dkahler_potential/biholomorphic/boolean_mask/strided_slice_2:output:0@kahler_potential/biholomorphic/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2kahler_potential/biholomorphic/boolean_mask/concat
3kahler_potential/biholomorphic/boolean_mask/ReshapeReshape,kahler_potential/biholomorphic/transpose:y:0;kahler_potential/biholomorphic/boolean_mask/concat:output:0*
T0*'
_output_shapes
:2џџџџџџџџџ25
3kahler_potential/biholomorphic/boolean_mask/ReshapeЭ
;kahler_potential/biholomorphic/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2=
;kahler_potential/biholomorphic/boolean_mask/Reshape_1/shape
5kahler_potential/biholomorphic/boolean_mask/Reshape_1Reshape/kahler_potential/biholomorphic/Squeeze:output:0Dkahler_potential/biholomorphic/boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:227
5kahler_potential/biholomorphic/boolean_mask/Reshape_1и
1kahler_potential/biholomorphic/boolean_mask/WhereWhere>kahler_potential/biholomorphic/boolean_mask/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ23
1kahler_potential/biholomorphic/boolean_mask/Whereѕ
3kahler_potential/biholomorphic/boolean_mask/SqueezeSqueeze9kahler_potential/biholomorphic/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
25
3kahler_potential/biholomorphic/boolean_mask/SqueezeИ
9kahler_potential/biholomorphic/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9kahler_potential/biholomorphic/boolean_mask/GatherV2/axis
4kahler_potential/biholomorphic/boolean_mask/GatherV2GatherV2<kahler_potential/biholomorphic/boolean_mask/Reshape:output:0<kahler_potential/biholomorphic/boolean_mask/Squeeze:output:0Bkahler_potential/biholomorphic/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ26
4kahler_potential/biholomorphic/boolean_mask/GatherV2Г
/kahler_potential/biholomorphic/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/kahler_potential/biholomorphic/transpose_1/perm
*kahler_potential/biholomorphic/transpose_1	Transpose=kahler_potential/biholomorphic/boolean_mask/GatherV2:output:08kahler_potential/biholomorphic/transpose_1/perm:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2,
*kahler_potential/biholomorphic/transpose_1в
,kahler_potential/dense/MatMul/ReadVariableOpReadVariableOp5kahler_potential_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,kahler_potential/dense/MatMul/ReadVariableOpр
kahler_potential/dense/MatMulMatMul.kahler_potential/biholomorphic/transpose_1:y:04kahler_potential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
kahler_potential/dense/MatMulЃ
kahler_potential/dense/SquareSquare'kahler_potential/dense/MatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
kahler_potential/dense/Squareи
.kahler_potential/dense_1/MatMul/ReadVariableOpReadVariableOp7kahler_potential_dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype020
.kahler_potential/dense_1/MatMul/ReadVariableOpй
kahler_potential/dense_1/MatMulMatMul!kahler_potential/dense/Square:y:06kahler_potential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
kahler_potential/dense_1/MatMulЉ
kahler_potential/dense_1/SquareSquare)kahler_potential/dense_1/MatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
kahler_potential/dense_1/Squareи
.kahler_potential/dense_2/MatMul/ReadVariableOpReadVariableOp7kahler_potential_dense_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype020
.kahler_potential/dense_2/MatMul/ReadVariableOpл
kahler_potential/dense_2/MatMulMatMul#kahler_potential/dense_1/Square:y:06kahler_potential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
kahler_potential/dense_2/MatMulЉ
kahler_potential/dense_2/SquareSquare)kahler_potential/dense_2/MatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
kahler_potential/dense_2/Square
&kahler_potential/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2(
&kahler_potential/Sum/reduction_indicesЗ
kahler_potential/SumSum#kahler_potential/dense_2/Square:y:0/kahler_potential/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
kahler_potential/Sum
kahler_potential/LogLogkahler_potential/Sum:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
kahler_potential/Logh
IdentityIdentitykahler_potential/Log:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ::::P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
С
p
*__inference_dense_2_layer_call_fn_19535502

inputs
unknown
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_195353582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџd:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Я9
g
K__inference_biholomorphic_layer_call_and_return_conditional_losses_19535452

inputs
identityE
ConjConjinputs*'
_output_shapes
:џџџџџџџџџ2
Conj
einsum/EinsumEinsuminputsConj:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equation
ai,aj->aij2
einsum/Einsumv
MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart/num_lower
MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2
MatrixBandPart/num_upperЦ
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:џџџџџџџџџ2
RealO
ImagImagReshape:output:0*'
_output_shapes
:џџџџџџџџџ2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ22
concatq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	Transposeconcat:output:0transpose/perm:output:0*
T0*'
_output_shapes
:2џџџџџџџџџ2
	transposeR
AbsAbstranspose:y:0*
T0*'
_output_shapes
:2џџџџџџџџџ2
Absp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices_
SumSumAbs:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes
:22
SumU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
Less/yX
LessLessSum:output:0Less/y:output:0*
T0*
_output_shapes
:22
LessL

LogicalNot
LogicalNotLess:z:0*
_output_shapes
:22

LogicalNotR
SqueezeSqueezeLogicalNot:y:0*
T0
*
_output_shapes
:22	
Squeezee
boolean_mask/ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stack
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_slice
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indicesЂ
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodi
boolean_mask/Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stack
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2И
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
boolean_mask/strided_slice_1i
boolean_mask/Shape_2Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_2
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stack
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2И
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2
boolean_mask/concat/values_1Packboolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat/values_1v
boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/concat/axisќ
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2џџџџџџџџџ2
boolean_mask/Reshape
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
boolean_mask/Reshape_1/shape
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ2
boolean_mask/Where
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisћ
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь

E__inference_dense_1_layer_call_and_return_conditional_losses_19535338

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџd::O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
ё

N__inference_kahler_potential_layer_call_and_return_conditional_losses_19535374
input_1
dense_19535327
dense_1_19535347
dense_2_19535367
identityЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallђ
biholomorphic/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_biholomorphic_layer_call_and_return_conditional_losses_195353022
biholomorphic/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_19535327*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_195353182
dense/StatefulPartitionedCallЄ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_19535347*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_195353382!
dense_1/StatefulPartitionedCallІ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_19535367*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_195353582!
dense_2/StatefulPartitionedCallp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices
SumSum(dense_2/StatefulPartitionedCall:output:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
SumM
LogLogSum:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
LogЛ
IdentityIdentityLog:y:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ:::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
ќ

C__inference_dense_layer_call_and_return_conditional_losses_19535318

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџ::X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ь

E__inference_dense_2_layer_call_and_return_conditional_losses_19535495

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџd::O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs

ь
$__inference__traced_restore_19535553
file_prefix
assignvariableop_variable!
assignvariableop_1_variable_1!
assignvariableop_2_variable_2

identity_4ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ђ
valueBB#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB#layer2/w/.ATTRIBUTES/VARIABLE_VALUEB#layer3/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
RestoreV2/shape_and_slicesП
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ђ
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ђ
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЅ

Identity_3Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_3

Identity_4IdentityIdentity_3:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2*
T0*
_output_shapes
: 2

Identity_4"!

identity_4Identity_4:output:0*!
_input_shapes
: :::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ь

E__inference_dense_1_layer_call_and_return_conditional_losses_19535480

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџd::O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs


3__inference_kahler_potential_layer_call_fn_19535386
input_1
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_kahler_potential_layer_call_and_return_conditional_losses_195353742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
ь

E__inference_dense_2_layer_call_and_return_conditional_losses_19535358

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџd::O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Я9
g
K__inference_biholomorphic_layer_call_and_return_conditional_losses_19535302

inputs
identityE
ConjConjinputs*'
_output_shapes
:џџџџџџџџџ2
Conj
einsum/EinsumEinsuminputsConj:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equation
ai,aj->aij2
einsum/Einsumv
MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart/num_lower
MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
џџџџџџџџџ2
MatrixBandPart/num_upperЦ
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:џџџџџџџџџ2
RealO
ImagImagReshape:output:0*'
_output_shapes
:џџџџџџџџџ2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ22
concatq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	Transposeconcat:output:0transpose/perm:output:0*
T0*'
_output_shapes
:2џџџџџџџџџ2
	transposeR
AbsAbstranspose:y:0*
T0*'
_output_shapes
:2џџџџџџџџџ2
Absp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices_
SumSumAbs:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes
:22
SumU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
Less/yX
LessLessSum:output:0Less/y:output:0*
T0*
_output_shapes
:22
LessL

LogicalNot
LogicalNotLess:z:0*
_output_shapes
:22

LogicalNotR
SqueezeSqueezeLogicalNot:y:0*
T0
*
_output_shapes
:22	
Squeezee
boolean_mask/ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stack
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_slice
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indicesЂ
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodi
boolean_mask/Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stack
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2И
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
boolean_mask/strided_slice_1i
boolean_mask/Shape_2Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_2
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stack
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2И
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2
boolean_mask/concat/values_1Packboolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat/values_1v
boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/concat/axisќ
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2џџџџџџџџџ2
boolean_mask/Reshape
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
boolean_mask/Reshape_1/shape
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ2
boolean_mask/Where
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisћ
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
L
0__inference_biholomorphic_layer_call_fn_19535457

inputs
identityе
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_biholomorphic_layer_call_and_return_conditional_losses_195353022
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С
p
*__inference_dense_1_layer_call_fn_19535487

inputs
unknown
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_195353382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџd:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Я
n
(__inference_dense_layer_call_fn_19535472

inputs
unknown
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_195353182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ї
serving_default
;
input_10
serving_default_input_1:0џџџџџџџџџ8
output_1,
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict: N

biholomorphic

layer1

layer2

layer3
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
*6&call_and_return_all_conditional_losses
7_default_save_signature
8__call__"
_tf_keras_modelі{"class_name": "KahlerPotential", "name": "kahler_potential", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "KahlerPotential"}}
Ф

trainable_variables
	variables
regularization_losses
	keras_api
*9&call_and_return_all_conditional_losses
:__call__"Е
_tf_keras_layer{"class_name": "Biholomorphic", "name": "biholomorphic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "biholomorphic", "trainable": true, "dtype": "float32"}}
Ѓ
w
trainable_variables
	variables
regularization_losses
	keras_api
*;&call_and_return_all_conditional_losses
<__call__"
_tf_keras_layerѓ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Ѕ
w
trainable_variables
	variables
regularization_losses
	keras_api
*=&call_and_return_all_conditional_losses
>__call__"
_tf_keras_layerѕ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Ѕ
w
trainable_variables
	variables
regularization_losses
	keras_api
*?&call_and_return_all_conditional_losses
@__call__"
_tf_keras_layerѕ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
layer_metrics

layers
metrics
trainable_variables
	variables
regularization_losses
 non_trainable_variables
!layer_regularization_losses
8__call__
7_default_save_signature
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
,
Aserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
"layer_metrics
#metrics

$layers

trainable_variables
	variables
regularization_losses
%non_trainable_variables
&layer_regularization_losses
:__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
:d2Variable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
'layer_metrics
(metrics

)layers
trainable_variables
	variables
regularization_losses
*non_trainable_variables
+layer_regularization_losses
<__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
:dd2Variable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
,layer_metrics
-metrics

.layers
trainable_variables
	variables
regularization_losses
/non_trainable_variables
0layer_regularization_losses
>__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
:dd2Variable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
1layer_metrics
2metrics

3layers
trainable_variables
	variables
regularization_losses
4non_trainable_variables
5layer_regularization_losses
@__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
<
0
1
2
3"
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
2
N__inference_kahler_potential_layer_call_and_return_conditional_losses_19535374Ц
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *&Ђ#
!
input_1џџџџџџџџџ
с2о
#__inference__wrapped_model_19535245Ж
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *&Ђ#
!
input_1џџџџџџџџџ
2ў
3__inference_kahler_potential_layer_call_fn_19535386Ц
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *&Ђ#
!
input_1џџџџџџџџџ
ѕ2ђ
K__inference_biholomorphic_layer_call_and_return_conditional_losses_19535452Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_biholomorphic_layer_call_fn_19535457Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_layer_call_and_return_conditional_losses_19535465Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_layer_call_fn_19535472Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_1_layer_call_and_return_conditional_losses_19535480Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_1_layer_call_fn_19535487Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_2_layer_call_and_return_conditional_losses_19535495Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_2_layer_call_fn_19535502Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
5B3
&__inference_signature_wrapper_19535399input_1
#__inference__wrapped_model_19535245h0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "/Њ,
*
output_1
output_1џџџџџџџџџА
K__inference_biholomorphic_layer_call_and_return_conditional_losses_19535452a/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 
0__inference_biholomorphic_layer_call_fn_19535457T/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!џџџџџџџџџџџџџџџџџџЄ
E__inference_dense_1_layer_call_and_return_conditional_losses_19535480[/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџd
 |
*__inference_dense_1_layer_call_fn_19535487N/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџdЄ
E__inference_dense_2_layer_call_and_return_conditional_losses_19535495[/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџd
 |
*__inference_dense_2_layer_call_fn_19535502N/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџdЋ
C__inference_dense_layer_call_and_return_conditional_losses_19535465d8Ђ5
.Ђ+
)&
inputsџџџџџџџџџџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџd
 
(__inference_dense_layer_call_fn_19535472W8Ђ5
.Ђ+
)&
inputsџџџџџџџџџџџџџџџџџџ
Њ "џџџџџџџџџdЌ
N__inference_kahler_potential_layer_call_and_return_conditional_losses_19535374Z0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "!Ђ

0џџџџџџџџџ
 
3__inference_kahler_potential_layer_call_fn_19535386M0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "џџџџџџџџџ
&__inference_signature_wrapper_19535399s;Ђ8
Ђ 
1Њ.
,
input_1!
input_1џџџџџџџџџ"/Њ,
*
output_1
output_1џџџџџџџџџ