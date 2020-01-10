#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkImageRegionIterator.h"

#include "itkExtractImageFilter.h"

#include "itkRescaleIntensityImageFilter.h"
#include "itkRecursiveGaussianImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"

#include "itkHistogram.h"

#include "itkDiscreteGaussianImageFilter.h"
#include "itkLaplacianImageFilter.h"

#include "itkZeroCrossingImageFilter.h"
#include "itkGrayscaleDilateImageFilter.h"
#include "itkBinaryCrossStructuringElement.h"
#include "itkCastImageFilter.h"

#include "itkBinaryThresholdImageFunction.h"
#include "itkFloodFilledImageFunctionConditionalIterator.h"

#include "itkMaskImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkLabelStatisticsImageFilter.h"

#include "itkChangeLabelImageFilter.h"

#include "itkVector.h"
#include "itkListSample.h"
#include "itkKdTree.h"
#include "itkWeightedCentroidKdTreeGenerator.h"
#include "itkKdTreeBasedKmeansEstimator.h"
#include "itkMinimumDecisionRule.h"
#include "itkEuclideanDistanceMetric.h" //changed
#include "itkSampleClassifierFilter.h"  //changed
#include "itkDecisionRule.h" //added
#include "itkMeasurementVectorTraits.h" //added
#include "itkClassifierBase.h" //added
#define vcl_fabs std::fabs //include to avoid errors

#include "itkDistanceToCentroidMembershipFunction.h"    //added
#include <iostream>

typedef float PixelType;
typedef unsigned long LabelPixelType;

typedef itk::Image<PixelType, 3> VolumeImageType;
typedef itk::Image<LabelPixelType, 3> LabeledVolumeImageType;

typedef itk::Image<PixelType, 2> SliceImageType;
typedef itk::Image<LabelPixelType, 2> LabeledSliceImageType;

typedef itk::ImageRegionIterator<VolumeImageType> VolumeIteratorType;
typedef itk::ImageRegionIterator<LabeledVolumeImageType> LabeledVolumeIteratorType;

typedef itk::ImageRegionIterator<SliceImageType> SliceIteratorType;
typedef itk::ImageRegionIterator<LabeledSliceImageType> LabeledSliceIteratorType;

const PixelType IntensityNormalizationThreshold = 100.0;
const double SigmaMinimum = 2.0;
const double SigmaMaximum = 4.0;
const unsigned int NumberOfSigmaSteps = 3;
const double PositiveLaplacianThresholdFactor = 0.2;


const unsigned int MeasurementVectorSize = 1;
const unsigned int NumberOfClasses = 2;

void KMeansClustering(std::vector<std::vector<PixelType> >& sampleVectors, std::vector<unsigned int>& classifiedVector)
{
    if (sampleVectors.size() != MeasurementVectorSize)
    {
        std::cout << "Error: sample vectors do not match measurement vector size." << std::endl;
        return;
    }

    if (sampleVectors[0].size() == 0)
    {
        return;
    }

    const unsigned int measurementVectorSize = MeasurementVectorSize;
    unsigned int numberOfClasses = NumberOfClasses;
    unsigned int sampleVectorSize = sampleVectors[0].size();

    typedef itk::Vector<double, MeasurementVectorSize> MeasurementVectorType;

    typedef itk::Statistics::ListSample<MeasurementVectorType> SampleType;
    SampleType::Pointer sample = SampleType::New();
    sample->SetMeasurementVectorSize(measurementVectorSize);

    MeasurementVectorType measurementVector;
    for (unsigned int i = 0; i < sampleVectorSize; i++)
    {
        for (unsigned int j = 0; j < measurementVectorSize; j++)
        {
            measurementVector[j] = sampleVectors[j][i];
        }
        sample->PushBack(measurementVector);
    }

    typedef itk::Statistics::WeightedCentroidKdTreeGenerator<SampleType> TreeGeneratorType;
    TreeGeneratorType::Pointer treeGenerator = TreeGeneratorType::New();

    treeGenerator->SetSample(sample);
    treeGenerator->SetBucketSize(16);
    treeGenerator->Update();

    typedef TreeGeneratorType::KdTreeType TreeType;
    typedef itk::Statistics::KdTreeBasedKmeansEstimator<TreeType> EstimatorType;
    EstimatorType::Pointer estimator = EstimatorType::New();

    EstimatorType::ParametersType initialMeans(numberOfClasses * measurementVectorSize);


    std::vector<unsigned int> classLabels;
    classLabels.resize(numberOfClasses);


    for (unsigned int i = 0; i < numberOfClasses; i++)
    {
        for (unsigned int j = 0; j < measurementVectorSize; j++)
        {
            initialMeans[i * measurementVectorSize + j] = (float)i;
        }
        classLabels[i] = i;
    }

    estimator->SetParameters(initialMeans);
    estimator->SetKdTree(treeGenerator->GetOutput());
    estimator->SetMaximumIteration(200);
    estimator->SetCentroidPositionChangesThreshold(0.0);
    estimator->StartOptimization();

    EstimatorType::ParametersType estimatedMeans = estimator->GetParameters();

    using MembershipFunctionType = itk::Statistics::DistanceToCentroidMembershipFunction< MeasurementVectorType >; //typedef itk::Statistics::EuclideanDistanceMetric<MeasurementVectorType> MembershipFunctionType;
    using MembershipFunctionPointer = MembershipFunctionType::Pointer; //aggiunto
    
    typedef itk::Statistics::MinimumDecisionRule DecisionRuleType;
    DecisionRuleType::Pointer decisionRule = DecisionRuleType::New();

    typedef itk::Statistics::SampleClassifierFilter<SampleType> ClassifierType;
    ClassifierType::Pointer classifier = ClassifierType::New();

    classifier->SetDecisionRule((itk::Statistics::DecisionRule::Pointer) decisionRule);
    classifier->SetInput(sample);
    classifier->SetNumberOfClasses(numberOfClasses);


    ///////
    typedef ClassifierType::ClassLabelVectorObjectType ClassLabelVectorObjectType;
    ClassLabelVectorObjectType::Pointer  classLabelsObject = ClassLabelVectorObjectType::New();
    typedef ClassifierType::ClassLabelVectorType ClassLabelVectorType;
    ClassLabelVectorType & classLabelsVector = classLabelsObject->Get();


    for (unsigned int i = 0; i < numberOfClasses; i++) {
        classLabelsVector.push_back(i); // classLabelsVector[i] = i;
    }
    ////////
    classifier->SetClassLabels(classLabelsObject);  //SetMembershipFunctions(classLabels) ????

    ////
    using MembershipFunctionVectorObjectType = ClassifierType::MembershipFunctionVectorObjectType;
    using MembershipFunctionVectorType = ClassifierType::MembershipFunctionVectorType;

    MembershipFunctionVectorObjectType::Pointer membershipFunctionVectorObject = MembershipFunctionVectorObjectType::New();
    MembershipFunctionVectorType & membershipFunctionVector = membershipFunctionVectorObject->Get();



    for (unsigned int i = 0; i < numberOfClasses; i++)
    {
        MembershipFunctionPointer membershipFunction = MembershipFunctionType::New(); //MembershipFunctionType::Pointer membershipFunction = MembershipFunctionType::New();
        MembershipFunctionType::CentroidType origin(sample->GetMeasurementVectorSize()); //OriginType

        for (unsigned int j = 0; j < measurementVectorSize; j++)
        {
            origin[j] = estimatedMeans[i * MeasurementVectorSize + j];
        }
        membershipFunction->SetCentroid(origin); //SetOrigin(origin);      
        membershipFunctionVector.push_back(membershipFunction.GetPointer());
    }
    classifier->SetMembershipFunctions(membershipFunctionVectorObject);
    classifier->Update();


    // Sort according to mean of component 0 of measurement vector
    unsigned int* classOrder = new unsigned int[numberOfClasses];
    for (unsigned int i = 0; i < numberOfClasses; i++)
    {
        classOrder[i] = i;
    }
    bool done = false;
    while (!done)
    {
        done = true;
        for (unsigned int i = 1; i < numberOfClasses; i++)
        {
            if (estimatedMeans[measurementVectorSize * classOrder[i]] < estimatedMeans[measurementVectorSize * classOrder[i - 1]])
            {
                unsigned int tmp = classOrder[i];
                classOrder[i] = classOrder[i - 1];
                classOrder[i - 1] = tmp;
                done = false;
            }
        }
    }

    const ClassifierType::MembershipSampleType* membershipSample = classifier->GetOutput(); //Output* e aggiunto const
    ClassifierType::MembershipSampleType::ConstIterator iter = membershipSample->Begin();
    while (iter != membershipSample->End())
    {
        unsigned int classLabel = iter.GetClassLabel();
        unsigned int outputClassLabel = classOrder[classLabel];
        classifiedVector.push_back(outputClassLabel);
        ++iter;
    }

    delete[] classOrder;
}


void LaplacianComputation(SliceImageType::Pointer inputSlice, SliceImageType::Pointer laplacianSlice)
{
    laplacianSlice->SetRegions(inputSlice->GetLargestPossibleRegion());
    laplacianSlice->Allocate();
    laplacianSlice->SetSpacing(inputSlice->GetSpacing());
    laplacianSlice->SetOrigin(inputSlice->GetOrigin());
    laplacianSlice->SetDirection(inputSlice->GetDirection());
    laplacianSlice->FillBuffer(itk::NumericTraits<PixelType>::Zero);

    typedef itk::DiscreteGaussianImageFilter<SliceImageType, SliceImageType> GaussianFilterType;
    GaussianFilterType::Pointer gaussianFilter = GaussianFilterType::New();
    gaussianFilter->SetInput(inputSlice);
    gaussianFilter->SetMaximumError(0.01);

    typedef itk::LaplacianImageFilter<SliceImageType, SliceImageType> LaplacianFilterType;
    LaplacianFilterType::Pointer laplacianFilter = LaplacianFilterType::New();
    laplacianFilter->SetInput(gaussianFilter->GetOutput());

    double sigmaStep = (SigmaMaximum - SigmaMinimum) / NumberOfSigmaSteps;

    for (unsigned int i = 0; i < NumberOfSigmaSteps; i++)
    {
        double sigma = SigmaMinimum + sigmaStep * i;
        gaussianFilter->SetVariance(sigma * sigma);

        try
        {
            laplacianFilter->Update();
        }
        catch (itk::ExceptionObject & ex)
        {
            std::cout << ex << std::endl;
            return;
        }

        SliceImageType::Pointer laplacianStepSlice = laplacianFilter->GetOutput();

        SliceIteratorType laplacianSliceIt(laplacianSlice, laplacianSlice->GetLargestPossibleRegion());
        SliceIteratorType laplacianStepSliceIt(laplacianStepSlice, laplacianStepSlice->GetLargestPossibleRegion());

        for (laplacianSliceIt.GoToBegin(), laplacianStepSliceIt.GoToBegin(); !laplacianSliceIt.IsAtEnd(); ++laplacianSliceIt, ++laplacianStepSliceIt)
        {
            PixelType laplacianValue = laplacianStepSliceIt.Get();
            PixelType currentLaplacianValue = laplacianSliceIt.Get();
            laplacianValue *= sigma;
            currentLaplacianValue *= sigma;
            if (vcl_fabs(laplacianValue) > vcl_fabs(currentLaplacianValue))
            {
                laplacianSliceIt.Set(laplacianValue);
            }
        }
    }
}

void EdgeDetection(SliceImageType::Pointer laplacianSlice, LabeledSliceImageType::Pointer edgeSlice)
{
    typedef itk::ZeroCrossingImageFilter<SliceImageType, LabeledSliceImageType> ZeroCrossingFilterType;
    ZeroCrossingFilterType::Pointer zeroCrossingFilter = ZeroCrossingFilterType::New();
    zeroCrossingFilter->SetInput(laplacianSlice);
    zeroCrossingFilter->SetForegroundValue(1);
    zeroCrossingFilter->SetBackgroundValue(0);
    try
    {
        zeroCrossingFilter->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cout << ex << std::endl;
        throw;
    }

    edgeSlice->Graft(zeroCrossingFilter->GetOutput());
}

void FloodFill(LabeledSliceImageType::Pointer edgeSlice, LabeledSliceImageType::Pointer floodFilledSlice)
{
    LabelPixelType backgroundLabel = 1;

    floodFilledSlice->SetRegions(edgeSlice->GetLargestPossibleRegion());
    floodFilledSlice->Allocate();
    floodFilledSlice->SetSpacing(edgeSlice->GetSpacing());
    floodFilledSlice->SetOrigin(edgeSlice->GetOrigin());
    floodFilledSlice->SetDirection(edgeSlice->GetDirection());

    typedef itk::ImageRegionIterator<LabeledSliceImageType> LabeledSliceIteratorType;
    LabeledSliceIteratorType floodFilledSliceIt(floodFilledSlice, floodFilledSlice->GetLargestPossibleRegion());
    for (floodFilledSliceIt.GoToBegin(); !floodFilledSliceIt.IsAtEnd(); ++floodFilledSliceIt)
    {
        floodFilledSliceIt.Set(0);
    }

    typedef itk::BinaryThresholdImageFunction<LabeledSliceImageType, double> FunctionType;
    typedef itk::FloodFilledImageFunctionConditionalIterator<LabeledSliceImageType, FunctionType> FloodFilledIteratorType;

    FunctionType::Pointer function = FunctionType::New();
    function->SetInputImage(edgeSlice);
    function->ThresholdBetween(0, backgroundLabel - 1);

    LabelPixelType currentLabel = 1;

    typedef SliceImageType::IndexType SliceIndexType;
    SliceIndexType zeroIndex;
    zeroIndex.Fill(0);
    std::vector<SliceIndexType> seedList;
    seedList.push_back(zeroIndex);

    function->SetInputImage(edgeSlice);
    function->ThresholdBetween(0, 0);

    FloodFilledIteratorType it(floodFilledSlice, function, seedList);
    it.GoToBegin();
    while (!it.IsAtEnd())
    {
        it.Set(currentLabel);
        ++it;
    }

    ++currentLabel;

    for (floodFilledSliceIt.GoToBegin(); !floodFilledSliceIt.IsAtEnd(); ++floodFilledSliceIt)
    {
        LabelPixelType label = floodFilledSliceIt.Get();
        if (label != 0)
        {
            continue;
        }
        if (edgeSlice->GetPixel(floodFilledSliceIt.GetIndex()) == backgroundLabel)
        {
            continue;
        }
        SliceIndexType labelIndex = floodFilledSliceIt.GetIndex();
        std::vector<SliceIndexType> seedList;
        seedList.push_back(labelIndex);
        FloodFilledIteratorType it(floodFilledSlice, function, seedList);
        it.GoToBegin();
        while (!it.IsAtEnd())
        {
            it.Set(currentLabel);
            ++it;
        }
        ++currentLabel;
    }

    for (floodFilledSliceIt.GoToBegin(); !floodFilledSliceIt.IsAtEnd(); ++floodFilledSliceIt)
    {
        LabelPixelType label = floodFilledSliceIt.Get();
        if (label > 0)
        {
            floodFilledSliceIt.Set(label - 1);
        }
    }

    typedef itk::BinaryCrossStructuringElement<PixelType, 2> KernelType;
    KernelType cross;
    KernelType::SizeType crossSize;
    crossSize[0] = 1;
    crossSize[1] = 1;
    cross.SetRadius(crossSize);
    cross.CreateStructuringElement();

    typedef itk::GrayscaleDilateImageFilter<LabeledSliceImageType, LabeledSliceImageType, KernelType> GrayscaleDilateFilterType;
    GrayscaleDilateFilterType::Pointer grayscaleDilateFilter = GrayscaleDilateFilterType::New();
    grayscaleDilateFilter->SetInput(floodFilledSlice);
    grayscaleDilateFilter->SetKernel(cross);
    try
    {
        grayscaleDilateFilter->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cout << ex << std::endl;
        throw;
    }

    floodFilledSlice->Graft(grayscaleDilateFilter->GetOutput());
}

void LaplacianCystSelection(SliceImageType::Pointer inputSlice, SliceImageType::Pointer laplacianSlice, LabeledSliceImageType::Pointer floodFilledSlice, LabeledSliceImageType::Pointer maskSlice, LabeledSliceImageType::Pointer labeledSlice)
{
    labeledSlice->SetRegions(inputSlice->GetLargestPossibleRegion());
    labeledSlice->Allocate();
    labeledSlice->SetSpacing(inputSlice->GetSpacing());
    labeledSlice->SetOrigin(inputSlice->GetOrigin());
    labeledSlice->SetDirection(inputSlice->GetDirection());

    typedef itk::MaskImageFilter<LabeledSliceImageType, LabeledSliceImageType> MaskFilterType;
    MaskFilterType::Pointer maskFilter = MaskFilterType::New();
    maskFilter->SetInput1(floodFilledSlice);
    maskFilter->SetInput2(maskSlice);
    try
    {
        maskFilter->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cerr << ex << std::endl;
    }

    typedef itk::RelabelComponentImageFilter<LabeledSliceImageType, LabeledSliceImageType> RelabelComponentFilterType;
    RelabelComponentFilterType::Pointer relabeler = RelabelComponentFilterType::New();
    relabeler->SetInput(maskFilter->GetOutput());
    try
    {
        relabeler->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cerr << ex << std::endl;
    }

    LabeledSliceImageType::Pointer maskedFloodFilledSlice = relabeler->GetOutput();

    typedef itk::LabelStatisticsImageFilter<SliceImageType, LabeledSliceImageType> LabelStatisticsFilterType;
    LabelStatisticsFilterType::Pointer labelStatistics = LabelStatisticsFilterType::New();
    labelStatistics->SetInput(laplacianSlice);
    labelStatistics->SetLabelInput(maskedFloodFilledSlice);
    try
    {
        labelStatistics->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cerr << ex << std::endl;
    }

    SliceIteratorType laplacianSliceIt(laplacianSlice, laplacianSlice->GetLargestPossibleRegion());
    LabeledSliceIteratorType maskSliceIt(maskSlice, maskSlice->GetLargestPossibleRegion());
    PixelType maximumPositiveLaplacian = itk::NumericTraits<PixelType>::Zero;
    for (laplacianSliceIt.GoToBegin(), maskSliceIt.GoToBegin(); !laplacianSliceIt.IsAtEnd(); ++laplacianSliceIt, ++maskSliceIt)
    {
        if (maskSliceIt.Get() == itk::NumericTraits<LabelPixelType>::Zero)
        {
            continue;
        }
        if (laplacianSliceIt.Get() > maximumPositiveLaplacian)
        {
            maximumPositiveLaplacian = laplacianSliceIt.Get();
        }
    }

    LabelPixelType numberOfLabels = static_cast<LabelPixelType>(labelStatistics->GetNumberOfLabels());

    std::vector<LabelPixelType> cystLabels;
    std::vector<LabelPixelType> backgroundLabels;

    for (LabelPixelType label = 0; label < numberOfLabels; label++)
    {
        PixelType meanLaplacian = labelStatistics->GetMean(label);
        PixelType maximumLaplacian = labelStatistics->GetMaximum(label);

        if (meanLaplacian > 0.0 && maximumLaplacian > PositiveLaplacianThresholdFactor* maximumPositiveLaplacian)
        {
            backgroundLabels.push_back(label);
            continue;
        }

        cystLabels.push_back(label);
    }

    typedef itk::ChangeLabelImageFilter<LabeledSliceImageType, LabeledSliceImageType> ChangeLabelFilterType;
    ChangeLabelFilterType::Pointer changeLabelFilter = ChangeLabelFilterType::New();
    changeLabelFilter->SetInput(maskedFloodFilledSlice);
    for (unsigned int i = 0; i < backgroundLabels.size(); i++)
    {
        changeLabelFilter->SetChange(backgroundLabels[i], itk::NumericTraits<LabelPixelType>::Zero);
    }
    try
    {
        changeLabelFilter->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cerr << ex << std::endl;
    }

    labeledSlice->Graft(changeLabelFilter->GetOutput());
}

void SliceProcessing(int sliceIndex, SliceImageType::Pointer inputSlice, LabeledSliceImageType::Pointer maskSlice, LabeledSliceImageType::Pointer floodFilledSlice, LabeledSliceImageType::Pointer labeledSlice)
{
    SliceImageType::Pointer laplacianSlice = SliceImageType::New();

    LaplacianComputation(inputSlice, laplacianSlice);

    LabeledSliceImageType::Pointer edgeSlice = LabeledSliceImageType::New();
    EdgeDetection(laplacianSlice, edgeSlice);

    FloodFill(edgeSlice, floodFilledSlice);

    LaplacianCystSelection(inputSlice, laplacianSlice, floodFilledSlice, maskSlice, labeledSlice);
}

void KMeansCystSelection(VolumeImageType::Pointer inputVolume, LabeledVolumeImageType::Pointer labeledVolume, LabeledVolumeImageType::Pointer outputLabeledVolume)
{
    outputLabeledVolume->SetRegions(inputVolume->GetLargestPossibleRegion());
    outputLabeledVolume->Allocate();
    outputLabeledVolume->SetSpacing(inputVolume->GetSpacing());
    outputLabeledVolume->SetOrigin(inputVolume->GetOrigin());
    outputLabeledVolume->SetDirection(inputVolume->GetDirection());

    LabeledVolumeIteratorType labeledVolumeIt(labeledVolume, labeledVolume->GetLargestPossibleRegion());
    LabeledVolumeIteratorType outputLabeledVolumeIt(outputLabeledVolume, outputLabeledVolume->GetLargestPossibleRegion());
    for (labeledVolumeIt.GoToBegin(), outputLabeledVolumeIt.GoToBegin(); !labeledVolumeIt.IsAtEnd(); ++labeledVolumeIt, ++outputLabeledVolumeIt)
    {
        outputLabeledVolumeIt.Set(labeledVolumeIt.Get());
    }

    typedef itk::LabelStatisticsImageFilter<VolumeImageType, LabeledVolumeImageType> LabelStatisticsFilterType;
    LabelStatisticsFilterType::Pointer labelStatistics = LabelStatisticsFilterType::New();
    labelStatistics->SetInput(inputVolume);
    labelStatistics->SetLabelInput(outputLabeledVolume);
    try
    {
        labelStatistics->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cerr << ex << std::endl;
    }

    LabelPixelType maxLabel = itk::NumericTraits<LabelPixelType>::Zero;
    for (outputLabeledVolumeIt.GoToBegin(); !outputLabeledVolumeIt.IsAtEnd(); ++outputLabeledVolumeIt)
    {
        LabelPixelType currentLabel = outputLabeledVolumeIt.Get();
        if (currentLabel > maxLabel)
        {
            maxLabel = currentLabel;
        }
    }

    std::vector<long> labelMap;
    labelMap.resize(maxLabel + 1);
    for (unsigned int i = 0; i < maxLabel + 1; i++)
    {
        labelMap[i] = -1;
    }

    std::vector<LabelPixelType> labelVector;
    for (outputLabeledVolumeIt.GoToBegin(); !outputLabeledVolumeIt.IsAtEnd(); ++outputLabeledVolumeIt)
    {
        LabelPixelType currentLabel = outputLabeledVolumeIt.Get();
        if (currentLabel == 0 || labelMap[currentLabel] != -1)
        {
            continue;
        }
        labelVector.push_back(currentLabel);
        labelMap[currentLabel] = labelVector.size() - 1;
    }

    if (labelVector.size() == 0)
    {
        return;
    }

    std::vector<PixelType> meanSampleVector;
    meanSampleVector.resize(labelVector.size());
    for (unsigned int i = 0; i < labelVector.size(); i++)
    {
        meanSampleVector[i] = labelStatistics->GetMean(labelVector[i]);
    }

    LabelPixelType backgroundLabel = 0;

    std::vector<unsigned int> classifiedVector;

    std::vector<std::vector<PixelType> > sampleVectors;
    sampleVectors.push_back(meanSampleVector);

    KMeansClustering(sampleVectors, classifiedVector);

    for (outputLabeledVolumeIt.GoToBegin(); !outputLabeledVolumeIt.IsAtEnd(); ++outputLabeledVolumeIt)
    {
        if (outputLabeledVolumeIt.Get() == backgroundLabel)
        {
            continue;
        }
        if (classifiedVector[labelMap[outputLabeledVolumeIt.Get()]] == 0)
        {
            outputLabeledVolumeIt.Set(backgroundLabel);
        }
    }
}

void SliceUniformityCorrection(int sliceIndex, SliceImageType::Pointer inputSlice, SliceImageType::Pointer processedSlice)
{
    double sigma = 50.0;

    typedef itk::RecursiveGaussianImageFilter<SliceImageType, SliceImageType> RecursiveGaussianFilterType;
    RecursiveGaussianFilterType::Pointer heavyGaussianFilterX = RecursiveGaussianFilterType::New();
    RecursiveGaussianFilterType::Pointer heavyGaussianFilterY = RecursiveGaussianFilterType::New();
    heavyGaussianFilterX->SetDirection(0);
    heavyGaussianFilterY->SetDirection(1);
    heavyGaussianFilterX->SetOrder(RecursiveGaussianFilterType::ZeroOrder);
    heavyGaussianFilterY->SetOrder(RecursiveGaussianFilterType::ZeroOrder);
    heavyGaussianFilterX->SetNormalizeAcrossScale(false);
    heavyGaussianFilterY->SetNormalizeAcrossScale(false);
    heavyGaussianFilterX->SetInput(inputSlice);
    heavyGaussianFilterY->SetInput(heavyGaussianFilterX->GetOutput());
    heavyGaussianFilterX->SetSigma(sigma);
    heavyGaussianFilterY->SetSigma(sigma);
    try
    {
        heavyGaussianFilterY->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cout << ex << std::endl;
        throw;
    }

    typedef itk::BinaryThresholdImageFilter<SliceImageType, SliceImageType> BinaryThresholdFilterType;
    BinaryThresholdFilterType::Pointer thresholdFilter = BinaryThresholdFilterType::New();
    thresholdFilter->SetInput(inputSlice);
    thresholdFilter->SetOutsideValue(1.0);
    thresholdFilter->SetInsideValue(500.0);
    thresholdFilter->SetLowerThreshold(IntensityNormalizationThreshold);
    try
    {
        thresholdFilter->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cout << ex << std::endl;
        throw;
    }

    RecursiveGaussianFilterType::Pointer binaryHeavyGaussianFilterX = RecursiveGaussianFilterType::New();
    RecursiveGaussianFilterType::Pointer binaryHeavyGaussianFilterY = RecursiveGaussianFilterType::New();
    binaryHeavyGaussianFilterX->SetDirection(0);
    binaryHeavyGaussianFilterY->SetDirection(1);
    binaryHeavyGaussianFilterX->SetOrder(RecursiveGaussianFilterType::ZeroOrder);
    binaryHeavyGaussianFilterY->SetOrder(RecursiveGaussianFilterType::ZeroOrder);
    binaryHeavyGaussianFilterX->SetNormalizeAcrossScale(false);
    binaryHeavyGaussianFilterY->SetNormalizeAcrossScale(false);
    binaryHeavyGaussianFilterX->SetInput(thresholdFilter->GetOutput());
    binaryHeavyGaussianFilterY->SetInput(binaryHeavyGaussianFilterX->GetOutput());
    binaryHeavyGaussianFilterX->SetSigma(sigma);
    binaryHeavyGaussianFilterY->SetSigma(sigma);
    try
    {
        binaryHeavyGaussianFilterY->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cout << ex << std::endl;
        throw;
    }

    processedSlice->SetRegions(inputSlice->GetLargestPossibleRegion());
    processedSlice->Allocate();
    processedSlice->SetSpacing(inputSlice->GetSpacing());
    processedSlice->SetOrigin(inputSlice->GetOrigin());
    processedSlice->SetDirection(inputSlice->GetDirection());

    SliceImageType::Pointer gaussianSlice = heavyGaussianFilterY->GetOutput();
    SliceImageType::Pointer binaryGaussianSlice = binaryHeavyGaussianFilterY->GetOutput();

    SliceIteratorType inputSliceIt(inputSlice, inputSlice->GetLargestPossibleRegion());
    SliceIteratorType gaussianSliceIt(gaussianSlice, gaussianSlice->GetLargestPossibleRegion());
    SliceIteratorType binaryGaussianSliceIt(binaryGaussianSlice, binaryGaussianSlice->GetLargestPossibleRegion());
    SliceIteratorType processedSliceIt(processedSlice, processedSlice->GetLargestPossibleRegion());

    const PixelType epsilon = 1E-8;
    for (inputSliceIt.GoToBegin(), gaussianSliceIt.GoToBegin(), binaryGaussianSliceIt.GoToBegin(), processedSliceIt.GoToBegin(); !inputSliceIt.IsAtEnd(); ++inputSliceIt, ++gaussianSliceIt, ++binaryGaussianSliceIt, ++processedSliceIt)
    {
        PixelType denominator = gaussianSliceIt.Get();
        if (vcl_fabs(denominator) < epsilon)
        {
            denominator = denominator > itk::NumericTraits<PixelType>::Zero ? 1.0 : -1.0;
        }
        PixelType outputValue = inputSliceIt.Get() * binaryGaussianSliceIt.Get() / denominator;
        processedSliceIt.Set(outputValue);
    }
}

void SliceIntensityNormalization(int sliceIndex, SliceImageType::Pointer inputSlice, LabeledSliceImageType::Pointer maskSlice, SliceImageType::Pointer processedSlice)
{
    processedSlice->SetRegions(inputSlice->GetLargestPossibleRegion());
    processedSlice->Allocate();
    processedSlice->SetSpacing(inputSlice->GetSpacing());
    processedSlice->SetOrigin(inputSlice->GetOrigin());
    processedSlice->SetDirection(inputSlice->GetDirection());

    typedef itk::Statistics::Histogram<PixelType> HistogramType; //,1

    PixelType minValue = itk::NumericTraits<PixelType>::max();
    PixelType maxValue = itk::NumericTraits<PixelType>::NonpositiveMin();

    SliceIteratorType inputIt(inputSlice, inputSlice->GetLargestPossibleRegion());
    SliceIteratorType processedIt(processedSlice, processedSlice->GetLargestPossibleRegion());
    LabeledSliceIteratorType maskIt(maskSlice, maskSlice->GetLargestPossibleRegion());

    for (inputIt.GoToBegin(), maskIt.GoToBegin(), processedIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt, ++maskIt, ++processedIt)
    {
        LabelPixelType maskLabel = maskIt.Get();
        if (maskLabel == 0)
        {
            continue;
        }
        PixelType value = inputIt.Get();

        minValue = value < minValue ? value : minValue;
        maxValue = value > maxValue ? value : maxValue;
    }

    HistogramType::Pointer histogram = HistogramType::New();

    HistogramType::SizeType size(1); //controllare qua (aggiungo 100) , risolvo assegnando 2
    size.Fill(100); //prova
    HistogramType::MeasurementVectorType lowerBound (1); //metto (numberOfComponents)
    HistogramType::MeasurementVectorType upperBound (1);
    //size[0] = 100; //problema su questa assegnazione
    lowerBound.Fill(minValue);
    upperBound.Fill(maxValue);
    
    histogram->SetMeasurementVectorSize(1); //added (numberOfComponents)

    histogram->Initialize(size, lowerBound, upperBound); //Controllare number of components e che valore riempire nel size
    histogram->SetToZero();

    HistogramType::MeasurementVectorType measurement (1); // (1)
    typedef HistogramType::MeasurementType MeasurementType;

    measurement[0] = itk::NumericTraits<MeasurementType>::Zero;

    for (inputIt.GoToBegin(), maskIt.GoToBegin(), processedIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt, ++maskIt, ++processedIt)
    {
        LabelPixelType maskLabel = maskIt.Get();
        if (maskLabel == 0)
        {
            continue;
        }
        PixelType value = inputIt.Get();
        measurement[0] = value;   
        histogram->IncreaseFrequencyOfMeasurement(measurement, 1); //IncreaseFrequency ???
    }

    const double lowerQuantile = 0.0;
    const double higherQuantile = 0.9;

    PixelType newMinValue = histogram->Quantile(0, lowerQuantile);
    PixelType newMaxValue = histogram->Quantile(0, higherQuantile);

    for (inputIt.GoToBegin(), maskIt.GoToBegin(), processedIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt, ++maskIt, ++processedIt)
    {
        PixelType value = inputIt.Get();
        PixelType factor = (value - newMinValue) / (newMaxValue - newMinValue);
        factor = factor < 0.0 ? 0.0 : factor;
        //factor = factor > 1.0 ? 1.0 : factor;
        PixelType newValue = minValue + factor * (maxValue - minValue);
        processedIt.Set(newValue);
    }
}

void NormalizeIntensity(VolumeImageType::Pointer nonNormalizedVolume, LabeledVolumeImageType::Pointer maskVolume, VolumeImageType::Pointer normalizedVolume)
{
    typedef itk::RescaleIntensityImageFilter<VolumeImageType, VolumeImageType> RescaleIntensityFilterType;
    RescaleIntensityFilterType::Pointer rescaleIntensity = RescaleIntensityFilterType::New();
    rescaleIntensity->SetOutputMinimum(1.0);
    rescaleIntensity->SetOutputMaximum(1000.0);
    rescaleIntensity->SetInput(nonNormalizedVolume);
    try
    {
        rescaleIntensity->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cout << ex << std::endl;
        return;
    }

    VolumeImageType::Pointer inputVolume = rescaleIntensity->GetOutput();
    VolumeImageType::RegionType inputRegion = inputVolume->GetLargestPossibleRegion();

    VolumeImageType::Pointer processedVolume = VolumeImageType::New();
    VolumeImageType::IndexType processedVolumeStart = inputRegion.GetIndex();
    VolumeImageType::SizeType processedVolumeSize = inputRegion.GetSize();
    VolumeImageType::RegionType processedVolumeRegion;
    processedVolumeRegion.SetSize(processedVolumeSize);
    processedVolumeRegion.SetIndex(processedVolumeStart);
    processedVolume->SetRegions(processedVolumeRegion);
    processedVolume->Allocate();
    processedVolume->SetSpacing(inputVolume->GetSpacing());
    processedVolume->SetOrigin(inputVolume->GetOrigin());
    processedVolume->SetDirection(inputVolume->GetDirection());

    typedef itk::ExtractImageFilter<VolumeImageType, SliceImageType> ExtractFilterType;
    ExtractFilterType::Pointer extractFilter = ExtractFilterType::New();

    typedef itk::ExtractImageFilter<LabeledVolumeImageType, LabeledSliceImageType> LabeledExtractFilterType;
    LabeledExtractFilterType::Pointer maskExtractFilter = LabeledExtractFilterType::New();

    VolumeImageType::SizeType size = inputRegion.GetSize();
    VolumeImageType::IndexType start = inputRegion.GetIndex();

    int maxSliceNumber = size[2];
    size[2] = 0;

    for (int k = start[2]; k < start[2] + maxSliceNumber; k++)
    {
        VolumeImageType::RegionType desiredRegion;
        VolumeImageType::IndexType desiredStart = start;
        desiredStart[2] = k;
        desiredRegion.SetSize(size);
        desiredRegion.SetIndex(desiredStart);

        extractFilter->SetExtractionRegion(desiredRegion);
        extractFilter->SetInput(inputVolume);
        extractFilter->SetDirectionCollapseToIdentity(); // This is required.
        try
        {
            extractFilter->Update();
        }
        catch (itk::ExceptionObject & ex)
        {
            std::cout << ex << std::endl;
            return;
        }

        maskExtractFilter->SetExtractionRegion(desiredRegion);
        maskExtractFilter->SetInput(maskVolume);
        maskExtractFilter->SetDirectionCollapseToIdentity(); // This is required.
        try
        {
            maskExtractFilter->Update();
        }
        catch (itk::ExceptionObject & ex)
        {
            std::cout << ex << std::endl;
            return;
        }

        SliceImageType::Pointer inputSlice = extractFilter->GetOutput();
        LabeledSliceImageType::Pointer maskSlice = maskExtractFilter->GetOutput();
        SliceImageType::Pointer uniformSlice = SliceImageType::New();

        try
        {
            SliceUniformityCorrection(k, inputSlice, uniformSlice);
        }
        catch (itk::ExceptionObject & ex)
        {
            std::cerr << ex << std::endl;
            return;
        }

        SliceImageType::Pointer processedSlice = SliceImageType::New();

        try
        {
            SliceIntensityNormalization(k, uniformSlice, maskSlice, processedSlice);

        }
        catch (itk::ExceptionObject & ex)
        {
            std::cerr << ex << std::endl;
            return;
        }

        processedVolumeStart[0] = 0;
        processedVolumeStart[1] = 0;
        processedVolumeStart[2] = k;
        processedVolumeSize[2] = 1;
        processedVolumeRegion.SetIndex(processedVolumeStart);
        processedVolumeRegion.SetSize(processedVolumeSize);

        SliceIteratorType processedSliceIt(processedSlice, processedSlice->GetLargestPossibleRegion());
        VolumeIteratorType processedVolumeIt(processedVolume, processedVolumeRegion);

        for (processedSliceIt.GoToBegin(), processedVolumeIt.GoToBegin(); !processedSliceIt.IsAtEnd(); ++processedSliceIt, ++processedVolumeIt)
        {
            processedVolumeIt.Set(processedSliceIt.Get());
        }
    }

    normalizedVolume->Graft(processedVolume);
}

int main()
{

    std::string path;
    std::cout << "Enter the directory that contains T2 and L,R.tif: ";
    std::cin >> path;




    typedef itk::ImageFileReader<VolumeImageType> VolumeReaderType;
    VolumeReaderType::Pointer volumeReader = VolumeReaderType::New();
    volumeReader->SetFileName(path + "//T2.mha");
    try
    {
        volumeReader->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cout << ex << std::endl;
        return EXIT_FAILURE;
    }

    typedef itk::ImageFileReader<LabeledVolumeImageType> LabeledVolumeReaderType;
    LabeledVolumeReaderType::Pointer maskVolumeReader = LabeledVolumeReaderType::New();
    maskVolumeReader->SetFileName(path + "//mask.mha");
    try
    {
        maskVolumeReader->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cout << ex << std::endl;
        return EXIT_FAILURE;
    }

    VolumeImageType::Pointer inputVolume = VolumeImageType::New();
    NormalizeIntensity(volumeReader->GetOutput(), maskVolumeReader->GetOutput(), inputVolume);

    VolumeImageType::RegionType inputRegion = inputVolume->GetLargestPossibleRegion();

    LabeledVolumeImageType::Pointer maskVolume = maskVolumeReader->GetOutput();

    LabeledVolumeImageType::Pointer floodFilledVolume = LabeledVolumeImageType::New();
    floodFilledVolume->SetRegions(inputRegion);
    floodFilledVolume->Allocate();
    floodFilledVolume->SetSpacing(inputVolume->GetSpacing());
    floodFilledVolume->SetOrigin(inputVolume->GetOrigin());
    floodFilledVolume->SetDirection(inputVolume->GetDirection());

    LabeledVolumeImageType::Pointer labeledVolume = LabeledVolumeImageType::New();
    labeledVolume->SetRegions(inputRegion);
    labeledVolume->Allocate();
    labeledVolume->SetSpacing(inputVolume->GetSpacing());
    labeledVolume->SetOrigin(inputVolume->GetOrigin());
    labeledVolume->SetDirection(inputVolume->GetDirection());

    typedef itk::ExtractImageFilter<VolumeImageType, SliceImageType> ExtractFilterType;
    ExtractFilterType::Pointer extractFilter = ExtractFilterType::New();

    typedef itk::ExtractImageFilter<LabeledVolumeImageType, LabeledSliceImageType> LabeledExtractFilterType;
    LabeledExtractFilterType::Pointer maskExtractFilter = LabeledExtractFilterType::New();

    VolumeImageType::SizeType size = inputRegion.GetSize();
    VolumeImageType::IndexType start = inputRegion.GetIndex();

    LabeledVolumeImageType::RegionType labeledVolumeRegion = labeledVolume->GetLargestPossibleRegion();
    LabeledVolumeImageType::RegionType::SizeType labeledVolumeSize = labeledVolumeRegion.GetSize();
    LabeledVolumeImageType::RegionType::IndexType labeledVolumeStart = labeledVolumeRegion.GetIndex();

    int maxSliceNumber = size[2];
    size[2] = 0;

    for (int k = start[2]; k < start[2] + maxSliceNumber; k++)
    {
        VolumeImageType::RegionType desiredRegion;
        VolumeImageType::IndexType desiredStart = start;
        desiredStart[2] = k;
        desiredRegion.SetSize(size);
        desiredRegion.SetIndex(desiredStart);

        extractFilter->SetExtractionRegion(desiredRegion);
        extractFilter->SetInput(inputVolume);
        extractFilter->SetDirectionCollapseToIdentity();
        try
        {
            extractFilter->Update();
        }
        catch (itk::ExceptionObject & ex)
        {
            std::cout << ex << std::endl;
            return EXIT_FAILURE;
        }

        maskExtractFilter->SetExtractionRegion(desiredRegion);
        maskExtractFilter->SetInput(maskVolume);
        maskExtractFilter->SetDirectionCollapseToIdentity();
        try
        {
            maskExtractFilter->Update();
        }
        catch (itk::ExceptionObject & ex)
        {
            std::cout << ex << std::endl;
            return EXIT_FAILURE;
        }

        SliceImageType::Pointer inputSlice = extractFilter->GetOutput();
        LabeledSliceImageType::Pointer maskSlice = maskExtractFilter->GetOutput();

        LabeledSliceImageType::Pointer floodFilledSlice = LabeledSliceImageType::New();
        LabeledSliceImageType::Pointer labeledSlice = LabeledSliceImageType::New();

        try
        {
            SliceProcessing(k, inputSlice, maskSlice, floodFilledSlice, labeledSlice);

        }
        catch (itk::ExceptionObject & ex)
        {
            std::cerr << ex << std::endl;
            return EXIT_FAILURE;
        }

        labeledVolumeStart[0] = 0;
        labeledVolumeStart[1] = 0;
        labeledVolumeStart[2] = k;
        labeledVolumeSize[2] = 1;
        labeledVolumeRegion.SetIndex(labeledVolumeStart);
        labeledVolumeRegion.SetSize(labeledVolumeSize);


        LabeledSliceIteratorType floodFilledSliceIt(floodFilledSlice, floodFilledSlice->GetLargestPossibleRegion());
        LabeledVolumeIteratorType floodFilledVolumeIt(floodFilledVolume, labeledVolumeRegion);
        for (floodFilledSliceIt.GoToBegin(), floodFilledVolumeIt.GoToBegin(); !floodFilledSliceIt.IsAtEnd(); ++floodFilledSliceIt, ++floodFilledVolumeIt)
        {
            floodFilledVolumeIt.Set(floodFilledSliceIt.Get());
        }

        LabeledSliceIteratorType labeledSliceIt(labeledSlice, labeledSlice->GetLargestPossibleRegion());
        LabeledVolumeIteratorType labeledVolumeIt(labeledVolume, labeledVolumeRegion);
        for (labeledSliceIt.GoToBegin(), labeledVolumeIt.GoToBegin(); !labeledSliceIt.IsAtEnd(); ++labeledSliceIt, ++labeledVolumeIt)
        {
            labeledVolumeIt.Set(labeledSliceIt.Get());
        }
    }
    

    size = inputRegion.GetSize();
    start = inputRegion.GetIndex();

    maxSliceNumber = size[2];
    size[2] = 1;


    LabelPixelType labelOffset = 0;
    for (int k = start[2]; k < start[2] + maxSliceNumber; k++)
    {
        int startSlice = k;
        VolumeImageType::RegionType sliceRegion;
        VolumeImageType::IndexType sliceStart = start;
        sliceStart[2] = startSlice;
        sliceRegion.SetSize(size);
        sliceRegion.SetIndex(sliceStart);

        LabelPixelType maxLabel, label, newLabel;
        maxLabel = 0;
        LabeledVolumeIteratorType labeledVolumeIt(labeledVolume, sliceRegion);
        for (labeledVolumeIt.GoToBegin(); !labeledVolumeIt.IsAtEnd(); ++labeledVolumeIt)
        {
            label = labeledVolumeIt.Get();
            if (label == 0)
            {
                continue;
            }
            newLabel = label + labelOffset;
            labeledVolumeIt.Set(newLabel);
            if (newLabel > maxLabel)
            {
                maxLabel = newLabel;
            }
        }
        if (maxLabel > labelOffset)
        {
            labelOffset = maxLabel;
        }
    }

    typedef itk::RelabelComponentImageFilter<LabeledVolumeImageType, LabeledVolumeImageType> RelabelComponentFilterType;
    RelabelComponentFilterType::Pointer relabeler = RelabelComponentFilterType::New();
    relabeler->SetInput(labeledVolume);
    try
    {
        relabeler->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cerr << ex << std::endl;
    }

    LabeledVolumeImageType::Pointer kMeansLabeledVolume = LabeledVolumeImageType::New();
    try
    {    
        KMeansCystSelection(inputVolume, labeledVolume, kMeansLabeledVolume);
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cerr << ex << std::endl;
        return EXIT_FAILURE;
    }

    typedef itk::ImageFileWriter<LabeledVolumeImageType> LabeledVolumeWriterType;
    LabeledVolumeWriterType::Pointer floodFilledVolumeWriter = LabeledVolumeWriterType::New();
    floodFilledVolumeWriter->SetFileName(path + "//T2_floodfilled.mha");
    floodFilledVolumeWriter->SetInput(floodFilledVolume);
    try
    {
        floodFilledVolumeWriter->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cerr << ex << std::endl;
        return EXIT_FAILURE;
    }

    LabeledVolumeWriterType::Pointer labeledVolumeWriter = LabeledVolumeWriterType::New();
    labeledVolumeWriter->SetFileName(path + "//T2_labeled.mha");
    labeledVolumeWriter->SetInput(kMeansLabeledVolume);
    try
    {
        labeledVolumeWriter->Update();
    }
    catch (itk::ExceptionObject & ex)
    {
        std::cerr << ex << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
           
}
