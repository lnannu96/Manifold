ƒб
—£
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
dtypetypeИ
Њ
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
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02v2.3.0-rc2-23-gb36436b0878ЕМ
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
q

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:	dи*
shared_name
Variable_2
j
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:	dи*
dtype0

NoOpNoOp
ў
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ф
valueКBЗ BА
Щ
biholomorphic

layer1

layer2

layer3
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
R

regularization_losses
	variables
trainable_variables
	keras_api
Y
w
regularization_losses
	variables
trainable_variables
	keras_api
Y
w
regularization_losses
	variables
trainable_variables
	keras_api
Y
w
regularization_losses
	variables
trainable_variables
	keras_api
 

0
1
2

0
1
2
≠
metrics
regularization_losses
layer_regularization_losses

layers
 layer_metrics
	variables
trainable_variables
!non_trainable_variables
 
 
 
 
≠
"metrics

regularization_losses
#layer_regularization_losses

$layers
%layer_metrics
	variables
trainable_variables
&non_trainable_variables
A?
VARIABLE_VALUEVariable#layer1/w/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
≠
'metrics
regularization_losses
(layer_regularization_losses

)layers
*layer_metrics
	variables
trainable_variables
+non_trainable_variables
CA
VARIABLE_VALUE
Variable_1#layer2/w/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
≠
,metrics
regularization_losses
-layer_regularization_losses

.layers
/layer_metrics
	variables
trainable_variables
0non_trainable_variables
CA
VARIABLE_VALUE
Variable_2#layer3/w/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
≠
1metrics
regularization_losses
2layer_regularization_losses

3layers
4layer_metrics
	variables
trainable_variables
5non_trainable_variables
 
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
z
serving_default_input_1Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Џ
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
:€€€€€€€€€*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_6736855
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
А
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
GPU2*0J 8В *)
f$R"
 __inference__traced_save_6736990
њ
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
GPU2*0J 8В *,
f'R%
#__inference__traced_restore_6737009Ро
л
З
D__inference_dense_1_layer_call_and_return_conditional_losses_6736794

inputs"
matmul_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:€€€€€€€€€d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:€€€€€€€€€d2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€d::O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
п
З
D__inference_dense_2_layer_call_and_return_conditional_losses_6736814

inputs"
matmul_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	dи*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€и2
MatMul_
SquareSquareMatMul:product:0*
T0*(
_output_shapes
:€€€€€€€€€и2
Square_
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€d::O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
Н
Ч
2__inference_kahler_potential_layer_call_fn_6736842
input_1
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_kahler_potential_layer_call_and_return_conditional_losses_67368302
StatefulPartitionedCallК
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
л
З
D__inference_dense_1_layer_call_and_return_conditional_losses_6736936

inputs"
matmul_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:€€€€€€€€€d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:€€€€€€€€€d2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€d::O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
Ѕ
o
)__inference_dense_2_layer_call_fn_6736958

inputs
unknown
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_67368142
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€d:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
≥
K
/__inference_biholomorphic_layer_call_fn_6736913

inputs
identity‘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_biholomorphic_layer_call_and_return_conditional_losses_67367582
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
з
О
M__inference_kahler_potential_layer_call_and_return_conditional_losses_6736830
input_1
dense_6736783
dense_1_6736803
dense_2_6736823
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallс
biholomorphic/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_biholomorphic_layer_call_and_return_conditional_losses_67367582
biholomorphic/PartitionedCallЪ
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_6736783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_67367742
dense/StatefulPartitionedCallҐ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_6736803*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_67367942!
dense_1/StatefulPartitionedCall•
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_6736823*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_67368142!
dense_2/StatefulPartitionedCallp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesЙ
SumSum(dense_2/StatefulPartitionedCall:output:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
SumM
LogLogSum:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
Logї
IdentityIdentityLog:y:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*#
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€:::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
ы
Е
B__inference_dense_layer_call_and_return_conditional_losses_6736774

inputs"
matmul_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:€€€€€€€€€d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:€€€€€€€€€d2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€€€€€€€€€€::X T
0
_output_shapes
:€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ќ
m
'__inference_dense_layer_call_fn_6736928

inputs
unknown
identityИҐStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_67367742
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€d2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€€€€€€€€€€:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ќ9
f
J__inference_biholomorphic_layer_call_and_return_conditional_losses_6736758

inputs
identityE
ConjConjinputs*'
_output_shapes
:€€€€€€€€€2
ConjШ
einsum/EinsumEinsuminputsConj:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€*
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
€€€€€€€€€2
MatrixBandPart/num_upper∆
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:€€€€€€€€€2
RealO
ImagImagReshape:output:0*'
_output_shapes
:€€€€€€€€€2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЛ
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€22
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
:2€€€€€€€€€2
	transposeR
AbsAbstranspose:y:0*
T0*'
_output_shapes
:2€€€€€€€€€2
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
 *oГ:2
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
boolean_mask/ShapeО
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stackТ
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1Т
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2Ь
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_sliceФ
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indicesҐ
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodi
boolean_mask/Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1Т
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stackЦ
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1Ц
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2Є
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
boolean_mask/Shape_2Т
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stackЦ
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1Ц
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2Є
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2О
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
boolean_mask/concat/axisь
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concatЦ
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2€€€€€€€€€2
boolean_mask/ReshapeП
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
boolean_mask/Reshape_1/shapeЩ
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:€€€€€€€€€2
boolean_mask/WhereШ
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:€€€€€€€€€*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisы
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permЭ
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ќ9
f
J__inference_biholomorphic_layer_call_and_return_conditional_losses_6736908

inputs
identityE
ConjConjinputs*'
_output_shapes
:€€€€€€€€€2
ConjШ
einsum/EinsumEinsuminputsConj:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€*
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
€€€€€€€€€2
MatrixBandPart/num_upper∆
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:€€€€€€€€€2
RealO
ImagImagReshape:output:0*'
_output_shapes
:€€€€€€€€€2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЛ
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€22
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
:2€€€€€€€€€2
	transposeR
AbsAbstranspose:y:0*
T0*'
_output_shapes
:2€€€€€€€€€2
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
 *oГ:2
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
boolean_mask/ShapeО
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stackТ
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1Т
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2Ь
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_sliceФ
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indicesҐ
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodi
boolean_mask/Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1Т
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stackЦ
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1Ц
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2Є
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
boolean_mask/Shape_2Т
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stackЦ
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1Ц
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2Є
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2О
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
boolean_mask/concat/axisь
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concatЦ
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2€€€€€€€€€2
boolean_mask/ReshapeП
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
boolean_mask/Reshape_1/shapeЩ
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:€€€€€€€€€2
boolean_mask/WhereШ
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:€€€€€€€€€*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisы
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permЭ
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
М
л
#__inference__traced_restore_6737009
file_prefix
assignvariableop_variable!
assignvariableop_1_variable_1!
assignvariableop_2_variable_2

identity_4ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_2Ц
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ґ
valueШBХB#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB#layer2/w/.ATTRIBUTES/VARIABLE_VALUEB#layer3/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЦ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
RestoreV2/shape_and_slicesњ
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

IdentityШ
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ґ
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ґ
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp•

Identity_3Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_3Ч

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
ы
Е
B__inference_dense_layer_call_and_return_conditional_losses_6736921

inputs"
matmul_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:€€€€€€€€€d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:€€€€€€€€€d2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€€€€€€€€€€::X T
0
_output_shapes
:€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ш
м
 __inference__traced_save_6736990
file_prefix'
#savev2_variable_read_readvariableop)
%savev2_variable_1_read_readvariableop)
%savev2_variable_2_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_237859ede2ea4a9c8f0a3d12ba65d621/part2	
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameР
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ґ
valueШBХB#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB#layer2/w/.ATTRIBUTES/VARIABLE_VALUEB#layer3/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesР
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
SaveV2/shape_and_slices∞
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*6
_input_shapes%
#: :d:dd:	dи: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d:$ 

_output_shapes

:dd:%!

_output_shapes
:	dи:

_output_shapes
: 
«q
ч
"__inference__wrapped_model_6736701
input_19
5kahler_potential_dense_matmul_readvariableop_resource;
7kahler_potential_dense_1_matmul_readvariableop_resource;
7kahler_potential_dense_2_matmul_readvariableop_resource
identityИД
#kahler_potential/biholomorphic/ConjConjinput_1*'
_output_shapes
:€€€€€€€€€2%
#kahler_potential/biholomorphic/Conjц
,kahler_potential/biholomorphic/einsum/EinsumEinsuminput_1,kahler_potential/biholomorphic/Conj:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€*
equation
ai,aj->aij2.
,kahler_potential/biholomorphic/einsum/Einsumі
7kahler_potential/biholomorphic/MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 29
7kahler_potential/biholomorphic/MatrixBandPart/num_lowerљ
7kahler_potential/biholomorphic/MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€29
7kahler_potential/biholomorphic/MatrixBandPart/num_upperб
-kahler_potential/biholomorphic/MatrixBandPartMatrixBandPart5kahler_potential/biholomorphic/einsum/Einsum:output:0@kahler_potential/biholomorphic/MatrixBandPart/num_lower:output:0@kahler_potential/biholomorphic/MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:€€€€€€€€€2/
-kahler_potential/biholomorphic/MatrixBandPart≠
,kahler_potential/biholomorphic/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2.
,kahler_potential/biholomorphic/Reshape/shapeъ
&kahler_potential/biholomorphic/ReshapeReshape4kahler_potential/biholomorphic/MatrixBandPart:band:05kahler_potential/biholomorphic/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2(
&kahler_potential/biholomorphic/Reshapeђ
#kahler_potential/biholomorphic/RealReal/kahler_potential/biholomorphic/Reshape:output:0*'
_output_shapes
:€€€€€€€€€2%
#kahler_potential/biholomorphic/Realђ
#kahler_potential/biholomorphic/ImagImag/kahler_potential/biholomorphic/Reshape:output:0*'
_output_shapes
:€€€€€€€€€2%
#kahler_potential/biholomorphic/ImagЪ
*kahler_potential/biholomorphic/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*kahler_potential/biholomorphic/concat/axis¶
%kahler_potential/biholomorphic/concatConcatV2,kahler_potential/biholomorphic/Real:output:0,kahler_potential/biholomorphic/Imag:output:03kahler_potential/biholomorphic/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€22'
%kahler_potential/biholomorphic/concatѓ
-kahler_potential/biholomorphic/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2/
-kahler_potential/biholomorphic/transpose/permы
(kahler_potential/biholomorphic/transpose	Transpose.kahler_potential/biholomorphic/concat:output:06kahler_potential/biholomorphic/transpose/perm:output:0*
T0*'
_output_shapes
:2€€€€€€€€€2*
(kahler_potential/biholomorphic/transposeѓ
"kahler_potential/biholomorphic/AbsAbs,kahler_potential/biholomorphic/transpose:y:0*
T0*'
_output_shapes
:2€€€€€€€€€2$
"kahler_potential/biholomorphic/AbsЃ
4kahler_potential/biholomorphic/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4kahler_potential/biholomorphic/Sum/reduction_indicesџ
"kahler_potential/biholomorphic/SumSum&kahler_potential/biholomorphic/Abs:y:0=kahler_potential/biholomorphic/Sum/reduction_indices:output:0*
T0*
_output_shapes
:22$
"kahler_potential/biholomorphic/SumУ
%kahler_potential/biholomorphic/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%kahler_potential/biholomorphic/Less/y‘
#kahler_potential/biholomorphic/LessLess+kahler_potential/biholomorphic/Sum:output:0.kahler_potential/biholomorphic/Less/y:output:0*
T0*
_output_shapes
:22%
#kahler_potential/biholomorphic/Less©
)kahler_potential/biholomorphic/LogicalNot
LogicalNot'kahler_potential/biholomorphic/Less:z:0*
_output_shapes
:22+
)kahler_potential/biholomorphic/LogicalNotѓ
&kahler_potential/biholomorphic/SqueezeSqueeze-kahler_potential/biholomorphic/LogicalNot:y:0*
T0
*
_output_shapes
:22(
&kahler_potential/biholomorphic/Squeeze¬
1kahler_potential/biholomorphic/boolean_mask/ShapeShape,kahler_potential/biholomorphic/transpose:y:0*
T0*
_output_shapes
:23
1kahler_potential/biholomorphic/boolean_mask/Shapeћ
?kahler_potential/biholomorphic/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?kahler_potential/biholomorphic/boolean_mask/strided_slice/stack–
Akahler_potential/biholomorphic/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Akahler_potential/biholomorphic/boolean_mask/strided_slice/stack_1–
Akahler_potential/biholomorphic/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Akahler_potential/biholomorphic/boolean_mask/strided_slice/stack_2÷
9kahler_potential/biholomorphic/boolean_mask/strided_sliceStridedSlice:kahler_potential/biholomorphic/boolean_mask/Shape:output:0Hkahler_potential/biholomorphic/boolean_mask/strided_slice/stack:output:0Jkahler_potential/biholomorphic/boolean_mask/strided_slice/stack_1:output:0Jkahler_potential/biholomorphic/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2;
9kahler_potential/biholomorphic/boolean_mask/strided_slice“
Bkahler_potential/biholomorphic/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bkahler_potential/biholomorphic/boolean_mask/Prod/reduction_indicesЮ
0kahler_potential/biholomorphic/boolean_mask/ProdProdBkahler_potential/biholomorphic/boolean_mask/strided_slice:output:0Kkahler_potential/biholomorphic/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 22
0kahler_potential/biholomorphic/boolean_mask/Prod∆
3kahler_potential/biholomorphic/boolean_mask/Shape_1Shape,kahler_potential/biholomorphic/transpose:y:0*
T0*
_output_shapes
:25
3kahler_potential/biholomorphic/boolean_mask/Shape_1–
Akahler_potential/biholomorphic/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Akahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack‘
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_1‘
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_2т
;kahler_potential/biholomorphic/boolean_mask/strided_slice_1StridedSlice<kahler_potential/biholomorphic/boolean_mask/Shape_1:output:0Jkahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack:output:0Lkahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_1:output:0Lkahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2=
;kahler_potential/biholomorphic/boolean_mask/strided_slice_1∆
3kahler_potential/biholomorphic/boolean_mask/Shape_2Shape,kahler_potential/biholomorphic/transpose:y:0*
T0*
_output_shapes
:25
3kahler_potential/biholomorphic/boolean_mask/Shape_2–
Akahler_potential/biholomorphic/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
Akahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack‘
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_1‘
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_2т
;kahler_potential/biholomorphic/boolean_mask/strided_slice_2StridedSlice<kahler_potential/biholomorphic/boolean_mask/Shape_2:output:0Jkahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack:output:0Lkahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_1:output:0Lkahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2=
;kahler_potential/biholomorphic/boolean_mask/strided_slice_2л
;kahler_potential/biholomorphic/boolean_mask/concat/values_1Pack9kahler_potential/biholomorphic/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2=
;kahler_potential/biholomorphic/boolean_mask/concat/values_1і
7kahler_potential/biholomorphic/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7kahler_potential/biholomorphic/boolean_mask/concat/axisґ
2kahler_potential/biholomorphic/boolean_mask/concatConcatV2Dkahler_potential/biholomorphic/boolean_mask/strided_slice_1:output:0Dkahler_potential/biholomorphic/boolean_mask/concat/values_1:output:0Dkahler_potential/biholomorphic/boolean_mask/strided_slice_2:output:0@kahler_potential/biholomorphic/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2kahler_potential/biholomorphic/boolean_mask/concatТ
3kahler_potential/biholomorphic/boolean_mask/ReshapeReshape,kahler_potential/biholomorphic/transpose:y:0;kahler_potential/biholomorphic/boolean_mask/concat:output:0*
T0*'
_output_shapes
:2€€€€€€€€€25
3kahler_potential/biholomorphic/boolean_mask/ReshapeЌ
;kahler_potential/biholomorphic/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2=
;kahler_potential/biholomorphic/boolean_mask/Reshape_1/shapeХ
5kahler_potential/biholomorphic/boolean_mask/Reshape_1Reshape/kahler_potential/biholomorphic/Squeeze:output:0Dkahler_potential/biholomorphic/boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:227
5kahler_potential/biholomorphic/boolean_mask/Reshape_1Ў
1kahler_potential/biholomorphic/boolean_mask/WhereWhere>kahler_potential/biholomorphic/boolean_mask/Reshape_1:output:0*'
_output_shapes
:€€€€€€€€€23
1kahler_potential/biholomorphic/boolean_mask/Whereх
3kahler_potential/biholomorphic/boolean_mask/SqueezeSqueeze9kahler_potential/biholomorphic/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:€€€€€€€€€*
squeeze_dims
25
3kahler_potential/biholomorphic/boolean_mask/SqueezeЄ
9kahler_potential/biholomorphic/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9kahler_potential/biholomorphic/boolean_mask/GatherV2/axisЦ
4kahler_potential/biholomorphic/boolean_mask/GatherV2GatherV2<kahler_potential/biholomorphic/boolean_mask/Reshape:output:0<kahler_potential/biholomorphic/boolean_mask/Squeeze:output:0Bkahler_potential/biholomorphic/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€26
4kahler_potential/biholomorphic/boolean_mask/GatherV2≥
/kahler_potential/biholomorphic/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/kahler_potential/biholomorphic/transpose_1/permЩ
*kahler_potential/biholomorphic/transpose_1	Transpose=kahler_potential/biholomorphic/boolean_mask/GatherV2:output:08kahler_potential/biholomorphic/transpose_1/perm:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2,
*kahler_potential/biholomorphic/transpose_1“
,kahler_potential/dense/MatMul/ReadVariableOpReadVariableOp5kahler_potential_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,kahler_potential/dense/MatMul/ReadVariableOpа
kahler_potential/dense/MatMulMatMul.kahler_potential/biholomorphic/transpose_1:y:04kahler_potential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
kahler_potential/dense/MatMul£
kahler_potential/dense/SquareSquare'kahler_potential/dense/MatMul:product:0*
T0*'
_output_shapes
:€€€€€€€€€d2
kahler_potential/dense/SquareЎ
.kahler_potential/dense_1/MatMul/ReadVariableOpReadVariableOp7kahler_potential_dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype020
.kahler_potential/dense_1/MatMul/ReadVariableOpў
kahler_potential/dense_1/MatMulMatMul!kahler_potential/dense/Square:y:06kahler_potential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2!
kahler_potential/dense_1/MatMul©
kahler_potential/dense_1/SquareSquare)kahler_potential/dense_1/MatMul:product:0*
T0*'
_output_shapes
:€€€€€€€€€d2!
kahler_potential/dense_1/Squareў
.kahler_potential/dense_2/MatMul/ReadVariableOpReadVariableOp7kahler_potential_dense_2_matmul_readvariableop_resource*
_output_shapes
:	dи*
dtype020
.kahler_potential/dense_2/MatMul/ReadVariableOp№
kahler_potential/dense_2/MatMulMatMul#kahler_potential/dense_1/Square:y:06kahler_potential/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€и2!
kahler_potential/dense_2/MatMul™
kahler_potential/dense_2/SquareSquare)kahler_potential/dense_2/MatMul:product:0*
T0*(
_output_shapes
:€€€€€€€€€и2!
kahler_potential/dense_2/SquareТ
&kahler_potential/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2(
&kahler_potential/Sum/reduction_indicesЈ
kahler_potential/SumSum#kahler_potential/dense_2/Square:y:0/kahler_potential/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
kahler_potential/SumА
kahler_potential/LogLogkahler_potential/Sum:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
kahler_potential/Logh
IdentityIdentitykahler_potential/Log:y:0*
T0*#
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€::::P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
њ
o
)__inference_dense_1_layer_call_fn_6736943

inputs
unknown
identityИҐStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_67367942
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€d2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€d:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
п
З
D__inference_dense_2_layer_call_and_return_conditional_losses_6736951

inputs"
matmul_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	dи*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€и2
MatMul_
SquareSquareMatMul:product:0*
T0*(
_output_shapes
:€€€€€€€€€и2
Square_
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€d::O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
’
К
%__inference_signature_wrapper_6736855
input_1
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_67367012
StatefulPartitionedCallК
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1"ЄL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*І
serving_defaultУ
;
input_10
serving_default_input_1:0€€€€€€€€€8
output_1,
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ЛN
Г
biholomorphic

layer1

layer2

layer3
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
*6&call_and_return_all_conditional_losses
7__call__
8_default_save_signature"Р
_tf_keras_modelц{"class_name": "KahlerPotential", "name": "kahler_potential", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "KahlerPotential"}}
ƒ

regularization_losses
	variables
trainable_variables
	keras_api
*9&call_and_return_all_conditional_losses
:__call__"µ
_tf_keras_layerЫ{"class_name": "Biholomorphic", "name": "biholomorphic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "biholomorphic", "trainable": true, "dtype": "float32"}}
£
w
regularization_losses
	variables
trainable_variables
	keras_api
*;&call_and_return_all_conditional_losses
<__call__"Н
_tf_keras_layerу{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
•
w
regularization_losses
	variables
trainable_variables
	keras_api
*=&call_and_return_all_conditional_losses
>__call__"П
_tf_keras_layerх{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
•
w
regularization_losses
	variables
trainable_variables
	keras_api
*?&call_and_return_all_conditional_losses
@__call__"П
_tf_keras_layerх{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
 "
trackable_list_wrapper
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
 
metrics
regularization_losses
layer_regularization_losses

layers
 layer_metrics
	variables
trainable_variables
!non_trainable_variables
7__call__
8_default_save_signature
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
≠
"metrics

regularization_losses
#layer_regularization_losses

$layers
%layer_metrics
	variables
trainable_variables
&non_trainable_variables
:__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
:d2Variable
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
≠
'metrics
regularization_losses
(layer_regularization_losses

)layers
*layer_metrics
	variables
trainable_variables
+non_trainable_variables
<__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
:dd2Variable
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
≠
,metrics
regularization_losses
-layer_regularization_losses

.layers
/layer_metrics
	variables
trainable_variables
0non_trainable_variables
>__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
:	dи2Variable
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
≠
1metrics
regularization_losses
2layer_regularization_losses

3layers
4layer_metrics
	variables
trainable_variables
5non_trainable_variables
@__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
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
Ы2Ш
M__inference_kahler_potential_layer_call_and_return_conditional_losses_6736830∆
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *&Ґ#
!К
input_1€€€€€€€€€
А2э
2__inference_kahler_potential_layer_call_fn_6736842∆
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *&Ґ#
!К
input_1€€€€€€€€€
а2Ё
"__inference__wrapped_model_6736701ґ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *&Ґ#
!К
input_1€€€€€€€€€
ф2с
J__inference_biholomorphic_layer_call_and_return_conditional_losses_6736908Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ў2÷
/__inference_biholomorphic_layer_call_fn_6736913Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_layer_call_and_return_conditional_losses_6736921Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_layer_call_fn_6736928Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_1_layer_call_and_return_conditional_losses_6736936Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_dense_1_layer_call_fn_6736943Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_2_layer_call_and_return_conditional_losses_6736951Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_dense_2_layer_call_fn_6736958Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
4B2
%__inference_signature_wrapper_6736855input_1О
"__inference__wrapped_model_6736701h0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "/™,
*
output_1К
output_1€€€€€€€€€ѓ
J__inference_biholomorphic_layer_call_and_return_conditional_losses_6736908a/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ З
/__inference_biholomorphic_layer_call_fn_6736913T/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "!К€€€€€€€€€€€€€€€€€€£
D__inference_dense_1_layer_call_and_return_conditional_losses_6736936[/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "%Ґ"
К
0€€€€€€€€€d
Ъ {
)__inference_dense_1_layer_call_fn_6736943N/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€d§
D__inference_dense_2_layer_call_and_return_conditional_losses_6736951\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "&Ґ#
К
0€€€€€€€€€и
Ъ |
)__inference_dense_2_layer_call_fn_6736958O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€и™
B__inference_dense_layer_call_and_return_conditional_losses_6736921d8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€d
Ъ В
'__inference_dense_layer_call_fn_6736928W8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€€€€€€€€€€
™ "К€€€€€€€€€dЂ
M__inference_kahler_potential_layer_call_and_return_conditional_losses_6736830Z0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "!Ґ
К
0€€€€€€€€€
Ъ Г
2__inference_kahler_potential_layer_call_fn_6736842M0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "К€€€€€€€€€Ь
%__inference_signature_wrapper_6736855s;Ґ8
Ґ 
1™.
,
input_1!К
input_1€€€€€€€€€"/™,
*
output_1К
output_1€€€€€€€€€