#include "itkImage.h"
#include "itkImageFileReader.h"

#include "itkImageRegionConstIterator.h"

int main()
{
  std::string path;
  std::cout<<"Enter the T2_labeled_ok path: ";
  std::cin>>path;

  const int Dim = 3;

  typedef unsigned long LabelPixelType;

  typedef itk::Image<LabelPixelType,Dim> LabeledImageType;

  typedef itk::ImageFileReader<LabeledImageType> LabelReaderType;
  LabelReaderType::Pointer labelReader = LabelReaderType::New();
  labelReader->SetFileName(path);
  try
  {
    labelReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  {
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }

  LabeledImageType::Pointer labeledImage = labelReader->GetOutput();

  LabeledImageType::SpacingType spacing = labeledImage->GetSpacing();

  double voxelVolume = spacing[0]*spacing[1]*spacing[2];

  typedef itk::ImageRegionConstIterator<LabeledImageType> LabeledImageIteratorType;

  LabeledImageIteratorType labeledIt(labeledImage,labeledImage->GetLargestPossibleRegion());

  unsigned int numberOfCystVoxels = 0;
  for(labeledIt.GoToBegin(); !labeledIt.IsAtEnd(); ++labeledIt)
    {
    if (labeledIt.Get() != itk::NumericTraits<LabelPixelType>::Zero)
      {
      ++numberOfCystVoxels;
      }
    }

  double cystVolume = numberOfCystVoxels * voxelVolume;
  std::cout<<path<<std::endl;
  std::cout<<"Volume: "<<cystVolume<<" mm^3"<<std::endl;
  std::cout<<"Volume: "<<cystVolume/1000<<" ml"<<std::endl;
  system("Pause");
}

