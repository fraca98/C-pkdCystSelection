#include <stdio.h>

#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"


int main()
{
  //pkdDicomSeriesToVolume
  std::string path;
  std::cout<<"Enter the directory that contains T2 and L,R.tif: ";
  std::cin>>path;

  const int Dim = 3;

  typedef float PixelType;
  typedef itk::Image<PixelType,Dim> ImageType;

  std::string dicomDirectoryName = path+"/T2";
  std::string outputImageFileName = path+"/T2.mha";

 // LEGGO LA SERIE DI IMMAGINI DICOM E CREO UNO STACK 3D

  std::cout<<"Reading image series"<<std::endl;
  typedef itk::ImageSeriesReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();

  typedef itk::GDCMImageIO ImageIOType;
  ImageIOType::Pointer dicomIO = ImageIOType::New();

  reader->SetImageIO(dicomIO);
  
  typedef itk::GDCMSeriesFileNames NamesGeneratorType;
  NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
  nameGenerator->SetUseSeriesDetails(true);
  nameGenerator->RecursiveOn();
  nameGenerator->SetDirectory(dicomDirectoryName);

  typedef std::vector<std::string> SeriesIdContainer;
  const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();

  SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
  SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
  while(seriesItr != seriesEnd)
    {
    std::cout << seriesItr->c_str() << std::endl;
    seriesItr++;
    }

  std::string seriesIdentifier;

  seriesIdentifier = seriesUID.begin()->c_str();

  typedef std::vector<std::string> FileNamesContainer;
  FileNamesContainer fileNames;

  fileNames = nameGenerator->GetFileNames(seriesIdentifier);

  reader->SetFileNames(fileNames);

  try
    {
    reader->Update();
    }
  catch (itk::ExceptionObject &ex)
    {
    std::cout << ex  << std::endl;
    return EXIT_FAILURE;
    }

// SCRIVO L'IMMAGINE 3D OTTENUTA (COMPRESSA) 

  typedef itk::ImageFileWriter< ImageType > WriterType; 
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(reader->GetOutput());
  writer->SetFileName(outputImageFileName.c_str());
  writer->UseCompressionOn();
  writer->Update();
  system("Pause");
}

