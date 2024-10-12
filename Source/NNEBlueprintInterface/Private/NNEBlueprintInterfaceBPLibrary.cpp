// Copyright Epic Games, Inc. All Rights Reserved.


#include "NNEBlueprintInterfaceBPLibrary.h"
#include "NNEBlueprintInterface.h"
#include "UObject/UObjectGlobals.h"

UNNEBlueprintInterfaceBPLibrary::UNNEBlueprintInterfaceBPLibrary(const FObjectInitializer& ObjectInitializer)
: Super(ObjectInitializer)
{

}

FNNTensorInfo::FNNTensorInfo(FTensorDesc desc)
{
	name = desc.GetName();
	dataType = desc.GetDataType();
	for (int32 sz : desc.GetShape().GetData())
	{
		shape.Add(sz);
	}
}
	
TArray<FString>  UNNEBlueprintInterfaceBPLibrary::GetRuntimeNames()
{
	return GetAllRuntimeNames();
}

FNNDataModel  UNNEBlueprintInterfaceBPLibrary::FromONNXFile(FString filePath, bool& success)
{
	TObjectPtr<UNNEModelData> ModelData =
		LoadObject<UNNEModelData>(NULL, *filePath);
	if (ModelData.IsNull())
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red,
			TEXT("Failed to load model data from file: " + filePath));
		success = false;
	}
	else
	{
		success = true;
	}
	return FNNDataModel(ModelData);
}

FNNDataModel UNNEBlueprintInterfaceBPLibrary::FromONNXBytes(TArray<uint8> byteArray, bool& success)
{
	TObjectPtr<UNNEModelData> ModelData = NewObject<UNNEModelData>();
	ModelData->Init("onnx", MakeArrayView(byteArray.GetData(), byteArray.Num()));
	if (ModelData.IsNull())
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red,
			TEXT("Failed to load model data from byte array"));
		success = false;
	}
	else
	{
		success = true;
	}
	return FNNDataModel(ModelData);
}




void  UNNEBlueprintInterfaceBPLibrary::CreateModelInstance(FNNDataModel modelData,FNNModelInstance& modelInstance,
                                                           bool& success)
{
	TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(FString("NNERuntimeORTCpu"));

	if (!Runtime.IsValid())
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red,
			TEXT("Could not fetch runtime NNERuntimeORTCpu " ));
		success = false;
		return;
	}
	
	const TObjectPtr<UNNEModelData> ModelData = modelData.ModelData;
	if (ModelData.IsNull())
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red,
			TEXT("Could not create ModelData " ));
		success = false;
		return;
	}

	if (!IsValid(ModelData))
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red,
			TEXT("Invalid ModelData " ));
		success = false;
		return;
	}
	
	const TSharedPtr<IModelCPU> Model = Runtime->CreateModelCPU(ModelData);
	if (!Model.IsValid())
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red,
			TEXT("Unable to create model"));
		success = false;
		return;
	}
	
	TSharedPtr<IModelInstanceCPU> ModelInstance = Model->CreateModelInstanceCPU();
	if (!ModelInstance.IsValid())
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red,
			TEXT("Unable to create ModelInstance " ));
		success = false;
		return;
	}
	modelInstance = FNNModelInstance(ModelInstance);
	success = true;
}

FNNIOInfo UNNEBlueprintInterfaceBPLibrary::GetModelIOInfo(FNNModelInstance modelInstance)
{
	TSharedPtr<IModelInstanceCPU> ModelInstance = modelInstance.ModelInstance;
	TConstArrayView<FTensorDesc> InputDescs = ModelInstance->GetInputTensorDescs();
	TConstArrayView<FTensorDesc> OutputDescs = ModelInstance->GetOutputTensorDescs();
	TArray<FNNTensorInfo> inputTensorInfo;
	TArray<FNNTensorInfo> outputTensorInfo;
	for (FTensorDesc desc : InputDescs)
	{
		inputTensorInfo.Add(FNNTensorInfo(desc));
	}
	for (FTensorDesc desc : OutputDescs)
	{
		outputTensorInfo.Add(FNNTensorInfo(desc));
	}
	return FNNIOInfo(inputTensorInfo, outputTensorInfo);
}
TArray<float> UNNEBlueprintInterfaceBPLibrary::RunModelInstance(FNNModelInstance modelInstance, TArray<float> inputTensorData, bool& success){
	TSharedPtr<IModelInstanceCPU> ModelInstance = modelInstance.ModelInstance;

	TArray<float> outputTensorData;
	TSharedPtr<IModelInstanceCPU> Model = modelInstance.ModelInstance;
	TArray<FTensorBindingCPU> InputBindings;
	TConstArrayView<FTensorBindingCPU> OutputBindings;
	for (float inputFloat : inputTensorData) //TODO make liek output if it works
	{
		FTensorBindingCPU InputBinding;
		InputBinding.Data = &inputFloat;
		InputBinding.SizeInBytes= sizeof(float);
		InputBindings.Add(InputBinding);
	}
	Model->RunSync(InputBindings,OutputBindings);
	for (FTensorBindingCPU outputBinding : OutputBindings)
	{
		float* outputElement = static_cast<float*>(outputBinding.Data);
		outputTensorData.Add(*outputElement);
	}
	success = true;
	return outputTensorData;
}







