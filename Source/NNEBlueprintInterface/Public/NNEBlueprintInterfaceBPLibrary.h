// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include <ObjectArray.h>

#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "NNEBlueprintInterfaceBPLibrary.generated.h"

using namespace UE::NNE;

// BP Wrapper for NNEModelData
USTRUCT(BlueprintType)
struct FNNDataModel
{
	GENERATED_BODY()
	UPROPERTY(BlueprintReadOnly)
	UNNEModelData* ModelData;
	
};

USTRUCT(BlueprintType)
struct FNNModelInstance
{
	GENERATED_BODY()
	
	TSharedPtr<IModelInstanceCPU> ModelInstance;
	FNNModelInstance() {}

	FNNModelInstance(TSharedPtr<IModelInstanceCPU> modelInstance)
	{
		ModelInstance = modelInstance;
	}
	
};
USTRUCT (BlueprintType)
struct FNNTensorInfo
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly)
	FString name = "";
	UPROPERTY(	BlueprintReadOnly)
	ENNETensorDataType dataType = ENNETensorDataType::None;
	UPROPERTY(BlueprintReadOnly)
	TArray<int32> shape = TArray<int32>();
	
	FNNTensorInfo(FTensorDesc desc);
	FNNTensorInfo() {}
	
};

USTRUCT(BlueprintType)
struct FNNIOInfo
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite)
	TArray<FNNTensorInfo> inputTensorInfo;
	UPROPERTY(BlueprintReadWrite)
	TArray<FNNTensorInfo> outputTensorInfo;

	FNNIOInfo(TArray<FNNTensorInfo>& inputTensorInfo, TArray<FNNTensorInfo>& outputTensorInfo)
		: inputTensorInfo(inputTensorInfo), outputTensorInfo(outputTensorInfo) {}
	FNNIOInfo()
	{
		inputTensorInfo = TArray<FNNTensorInfo>();
		outputTensorInfo = TArray<FNNTensorInfo>();
	}
};




/* 
*	Function library class.
*	Each function in it is expected to be static and represents blueprint node that can be called in any blueprint.
*
*	When declaring function you can define metadata for the node. Key function specifiers will be BlueprintPure and BlueprintCallable.
*	BlueprintPure - means the function does not affect the owning object in any way and thus creates a node without Exec pins.
*	BlueprintCallable - makes a function which can be executed in Blueprints - Thus it has Exec pins.
*	DisplayName - full name of the node, shown when you mouse over the node and in the blueprint drop down menu.
*				Its lets you name the node using characters not allowed in C++ function names.
*	CompactNodeTitle - the word(s) that appear on the node.
*	Keywords -	the list of keywords that helps you to find node when you search for it using Blueprint drop-down menu. 
*				Good example is "Print String" node which you can find also by using keyword "log".
*	Category -	the category your node will be under in the Blueprint drop-down menu.
*
*	For more info on custom blueprint nodes visit documentation:
*	https://wiki.unrealengine.com/Custom_Blueprint_Node_Creation
*/
UCLASS()
class UNNEBlueprintInterfaceBPLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_UCLASS_BODY()

	UFUNCTION(BlueprintCallable,
			meta = (DisplayName = "Get NNE Runtime Names", Keywords = "NNEInterface NNE Model"),
			Category = "NNEInterface")
	TArray<FString> GetRuntimeNames();

	UFUNCTION(BlueprintCallable,
		meta = (DisplayName = "Load Model From File", Keywords = "NNEInterface NNE Model"),
		Category = "NNEInterface")
	static FNNDataModel FromONNXFile( FString filePath, bool& success);

	UFUNCTION(BlueprintCallable,
		meta = (DisplayName = "Load Model From ONNX bytes", Keywords = "NNEInterface NNE Model"),
		Category = "NNEInterface")
	static FNNDataModel FromONNXBytes( TArray<uint8> byteArray, bool& success);

		
	UFUNCTION(BlueprintCallable,
		meta = (DisplayName = "Create NN model from ModelData", Keywords = "NNEInterface NNE Model"),
		Category = "NNEInterface")
	static void CreateModelInstance(FNNDataModel modelData, FNNModelInstance& modelInstance, bool& success);

	UFUNCTION(BlueprintCallable,
		meta = (DisplayName = "Get Model IO info", Keywords = "NNEInterface NNE Model"),
		Category = "NNEInterface")
	static FNNIOInfo GetModelIOInfo(FNNModelInstance modelInstance);

	UFUNCTION(BlueprintCallable,
		meta = (DisplayName = "Run Model Instance", Keywords = "NNEInterface NNE Model"),
		Category = "NNEInterface")
	static TArray<float> RunModelInstance(FNNModelInstance modelInstance, TArray<float> inputTensorData, bool& success);


	
	
};
