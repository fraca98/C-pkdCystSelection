#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkImageRegionIterator.h"
#include "itkMaximumImageFilter.h"
#include "itkFlipImageFilter.h"



int main()
{
  std::string path;
  std::cout<<"Enter the directory that contains T2 and L,R.tif: ";
  std::cin>>path;
  const int Dim = 3;

  typedef float PixelType;
  typedef unsigned long LabelPixelType;

  typedef itk::Image<PixelType,Dim> ImageType;
  typedef itk::Image<LabelPixelType,Dim> LabeledImageType;

  typedef unsigned char MaskPixelType;
  typedef itk::Image<MaskPixelType,Dim> MaskImageType;

// read the 4mm input volume
  typedef itk::ImageFileReader<LabeledImageType> LabelReaderType;
  LabelReaderType::Pointer labelReader = LabelReaderType::New();
  labelReader->SetFileName(path+"//T2.mha");
  try
  {
    labelReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  {
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }

// read the 4mm T2
  typedef itk::ImageFileReader<ImageType> FileReaderType;
  FileReaderType::Pointer reader4mm = FileReaderType::New();
  reader4mm->SetFileName(path+"//T2.mha");
  try
  {
    reader4mm->Update();
  }
  catch (itk::ExceptionObject &ex)
  {
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }

// define outputimage for the iterator

  LabeledImageType::Pointer outputImage = labelReader->GetOutput();

// read the masks and compute an overall mask

  std::cout<<"Reading mask1"<<std::endl;
  typedef itk::ImageFileReader<MaskImageType> MaskImageFileReaderType;
  MaskImageFileReaderType::Pointer maskReader1 = MaskImageFileReaderType::New();
  maskReader1->SetFileName(path+"//L.tif");
  maskReader1->Update();

  std::cout<<"Reading mask2"<<std::endl;
  MaskImageFileReaderType::Pointer maskReader2 = MaskImageFileReaderType::New();
  maskReader2->SetFileName(path+"//R.tif");
  maskReader2->Update();

  typedef itk::MaximumImageFilter< MaskImageType, MaskImageType, MaskImageType > MaximumFilterType;
  MaximumFilterType::Pointer maximumFilter = MaximumFilterType::New();
  maximumFilter->SetInput1(maskReader1->GetOutput() );
  maximumFilter->SetInput2(maskReader2->GetOutput() );
  try
    {
      maximumFilter->Update();
    }
  catch(itk::ExceptionObject & exp)
    {
      std::cerr << exp << std::endl;
    }

  MaskImageType::Pointer maskImage = maximumFilter->GetOutput();

// eventually flip the mask in the z direction

  int flipInput = 1;

  if (flipInput)
    {
    typedef itk::FlipImageFilter<MaskImageType> FlipImageFilterType;
    FlipImageFilterType::Pointer flipFilter = FlipImageFilterType::New();
    typedef FlipImageFilterType::FlipAxesArrayType FlipAxesArrayType;
    FlipAxesArrayType flipArray;
    flipArray[0] = 0;
    flipArray[1] = 0;
    flipArray[2] = flipInput;
    flipFilter->SetFlipAxes(flipArray);
    flipFilter->SetInput(maximumFilter->GetOutput());
    flipFilter->FlipAboutOriginOff();
    try
      {
        flipFilter->Update();
      }
    catch(itk::ExceptionObject & exp)
      {
        std::cerr << exp << std::endl;
      }

    maskImage = flipFilter->GetOutput();
    }

  maskImage->SetOrigin(reader4mm->GetOutput()->GetOrigin());
  maskImage->SetSpacing(reader4mm->GetOutput()->GetSpacing());
  maskImage->SetDirection(reader4mm->GetOutput()->GetDirection());

  // define an interpolator running on the mask to interpolate the mask to spacing if there are differents between mm

  typedef itk::NearestNeighborInterpolateImageFunction<MaskImageType,double> MaskInterpolateType;

  MaskInterpolateType::Pointer maskInterpolator = MaskInterpolateType::New();
  maskInterpolator->SetInputImage(maskImage);

  // use an iterator running on the 4mm volume to interpolate the mask

  typedef itk::ImageRegionIterator<LabeledImageType> IteratorType;  
  IteratorType maskIt(outputImage,outputImage->GetLargestPossibleRegion());

  LabelPixelType backgroundValue = 0;
  LabelPixelType foregroundValue = 1;

  for (maskIt.GoToBegin(); !maskIt.IsAtEnd(); ++maskIt)
  {
      ImageType::PointType point;

      outputImage->TransformIndexToPhysicalPoint(maskIt.GetIndex(), point);
      if (maskInterpolator->IsInsideBuffer(point))
      {
          if (maskInterpolator->Evaluate(point) == 0)
          {
              maskIt.Set(backgroundValue);
          }
          else
          {
              maskIt.Set(foregroundValue);
          }
      }
      else
      {
          maskIt.Set(backgroundValue);
      }
  }

// write the mask output volume
  typedef itk::ImageFileWriter<LabeledImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(path+"//mask.mha");
  writer->SetInput(outputImage);
  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject &ex)
  {
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

