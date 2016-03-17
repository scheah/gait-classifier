import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Path;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class TimeFreqDomainClassifier extends DataClassifier {

	public TimeFreqDomainClassifier(String[] classes, int classifierOption) {
		super(classes, classifierOption);
	}

	protected Instances initializeTransformedDataFormat() {
		int numAttributes = 28;
		String[] axis = new String[] { "x", "y", "z" };
		String[] classes = m_Classes;

		// Transformed Data
		FastVector fvTransformedWekaAttributes = new FastVector(numAttributes);

		// Numeric attributes

		// Mean
		for (int i = 0; i < axis.length; i++) {
			Attribute attribute = new Attribute("Mean " + axis[i]);
			fvTransformedWekaAttributes.addElement(attribute);
		}

		// Variance
		for (int i = 0; i < axis.length; i++) {
			Attribute attribute = new Attribute("Variance " + axis[i]);
			fvTransformedWekaAttributes.addElement(attribute);
		}

		// Standard Deviation
		for (int i = 0; i < axis.length; i++) {
			Attribute attribute = new Attribute("Standard Deviation " + axis[i]);
			fvTransformedWekaAttributes.addElement(attribute);
		}

		// Minimum
		for (int i = 0; i < axis.length; i++) {
			Attribute attribute = new Attribute("Minimum " + axis[i]);
			fvTransformedWekaAttributes.addElement(attribute);
		}

		// Average Absolute Sample Difference
		for (int i = 0; i < axis.length; i++) {
			Attribute attribute = new Attribute("Average Absolute Sample Difference " + axis[i]);
			fvTransformedWekaAttributes.addElement(attribute);
		}

		// Correlation
		for (int i = 0; i < axis.length - 1; i++) {
			for (int j = i + 1; j < axis.length; j++) {
				Attribute attribute = new Attribute("Correlation " + axis[i] + axis[j]);
				fvTransformedWekaAttributes.addElement(attribute);
			}
		}

		// Frequency Data

		// Sepctral Energy
		for (int i = 0; i < axis.length; i++) {
			Attribute attribute = new Attribute("Spectral Energy " + axis[i]);
			fvTransformedWekaAttributes.addElement(attribute);
		}

		// Fourier Coefficients
		for (int i = 0; i < axis.length; i++) {
			Attribute attribute = new Attribute("Fourier Coeff " + axis[i]);
			fvTransformedWekaAttributes.addElement(attribute);
		}

		// Sepctral Entropy
		for (int i = 0; i < axis.length; i++) {
			Attribute attribute = new Attribute("Spectral Entropy " + axis[i]);
			fvTransformedWekaAttributes.addElement(attribute);
		}

		// Declare class attribute
		FastVector fvUserVal = new FastVector(classes.length);
		for (int i = 0; i < classes.length; i++) {
			fvUserVal.addElement(classes[i]);
		}
		Attribute classAttribute = new Attribute("userid", fvUserVal);
		fvTransformedWekaAttributes.addElement(classAttribute);

		// Create the instances
		Instances transformedData = new Instances("FeaturesV2", fvTransformedWekaAttributes, 25);
		transformedData.setClassIndex(transformedData.numAttributes() - 1);
		return transformedData;
	}

	protected Instance transformData(Instances rawData, Instances rawSpectralData, String userid) {
		String[] axis = new String[] { "x", "y", "z" };
		Instance windowDataPoint = new DenseInstance(28);

		// Mean, Variance, Standard Deviation, Minimum
		for (int i = 0; i < axis.length; i++) {
			double mean = rawData.meanOrMode(i);
			double var = rawData.variance(i);
			double sd = Math.sqrt(var);
			double minimum = rawData.attributeStats(i).numericStats.min;
			windowDataPoint.setValue(m_Data.attribute("Mean " + axis[i]), mean);
			windowDataPoint.setValue(m_Data.attribute("Variance " + axis[i]), var);
			windowDataPoint.setValue(m_Data.attribute("Standard Deviation " + axis[i]), sd);
			windowDataPoint.setValue(m_Data.attribute("Minimum " + axis[i]), minimum);
		}

		// Average Absolute Sample Difference
		double[] sumDeltas = new double[] { 0, 0, 0 };
		Instance prevInst = rawData.instance(0);
		for (int instIdx = 1; instIdx < rawData.numInstances(); instIdx++) {
			Instance currInst = rawData.instance(instIdx);
			for (int i = 0; i < axis.length; i++) {
				sumDeltas[i] += currInst.value(rawData.attribute(axis[i])) - prevInst.value(rawData.attribute(axis[i]));
			}
			prevInst = currInst;
		}
		for (int i = 0; i < axis.length; i++) {
			windowDataPoint.setValue(m_Data.attribute("Average Absolute Sample Difference " + axis[i]),
					sumDeltas[i] / rawData.numInstances());
		}
		for (int i = 0; i < axis.length - 1; i++) {
			for (int j = i + 1; j < axis.length; j++) {
				double[] arrI = rawData.attributeToDoubleArray(i);
				double[] arrJ = rawData.attributeToDoubleArray(j);
				windowDataPoint.setValue(m_Data.attribute("Correlation " + axis[i] + axis[j]),
						Utils.correlation(arrI, arrJ, arrI.length));
			}
		}
		for (int i = 0; i < axis.length; i++) {
			double value = rawSpectralData.meanOrMode(i);
			windowDataPoint.setValue(m_Data.attribute("Spectral Energy " + axis[i]), value);
		}
		for (int i = 0; i < axis.length; i++) {
			double value = rawSpectralData.meanOrMode(i+3);
			windowDataPoint.setValue(m_Data.attribute("Fourier Coeff " + axis[i]), value);
		}
		for (int i = 0; i < axis.length; i++) {
			double value = rawSpectralData.meanOrMode(i+6);
			windowDataPoint.setValue(m_Data.attribute("Spectral Entropy " + axis[i]), value);
		}
		// Class attribute
		windowDataPoint.setValue(m_Data.attribute("userid"), userid);

		return windowDataPoint;
	}

	protected void readData(File file, String userid) throws Exception {
		Instances rawData = initializeRawDataFormat();

		Attribute attribute1 = new Attribute("spectral_energy_x");
		Attribute attribute2 = new Attribute("spectral_energy_y");
		Attribute attribute3 = new Attribute("spectral_energy_z");
		Attribute attribute4 = new Attribute("fourier_coeff_x");
		Attribute attribute5 = new Attribute("fourier_coeff_y");
		Attribute attribute6 = new Attribute("fourier_coeff_z");
		Attribute attribute7 = new Attribute("spectral_entropy_x");
		Attribute attribute8 = new Attribute("spectral_entropy_y");
		Attribute attribute9 = new Attribute("spectral_entropy_z");
		FastVector fvWekaAttributes = new FastVector(9);
		fvWekaAttributes.addElement(attribute1);
		fvWekaAttributes.addElement(attribute2);
		fvWekaAttributes.addElement(attribute3);
		fvWekaAttributes.addElement(attribute4);
		fvWekaAttributes.addElement(attribute5);
		fvWekaAttributes.addElement(attribute6);
		fvWekaAttributes.addElement(attribute7);
		fvWekaAttributes.addElement(attribute8);
		fvWekaAttributes.addElement(attribute9);
		Instances rawSpectralData = new Instances("raw_spectral", fvWekaAttributes, 250);

		if (file == null)
			return;

		BufferedReader reader = new BufferedReader(new FileReader(file));
		String mLine;
		while ((mLine = reader.readLine()) != null) {
			if (mLine.contains(":") || mLine.contains("="))
				continue;
			String[] data = mLine.split("\\s+");
			Instance iDataPoint = new DenseInstance(3);
			iDataPoint.setValue(rawData.attribute("x"), Double.parseDouble(data[0]));
			iDataPoint.setValue(rawData.attribute("y"), Double.parseDouble(data[1]));
			iDataPoint.setValue(rawData.attribute("z"), Double.parseDouble(data[2]));
			rawData.add(iDataPoint);
		}
		reader.close();

		Path path = file.toPath();
		String filename = "spectral_energy_" + path.getFileName().toString();
		File spectralFile = new File(path.getParent().toString() + "/" + filename);

		reader = new BufferedReader(new FileReader(spectralFile));
		while ((mLine = reader.readLine()) != null) {
			if (mLine.contains(":") || mLine.contains("="))
				continue;
			String[] data = mLine.split("\\s+");
			Instance iDataPoint = new DenseInstance(9);
			iDataPoint.setValue(rawSpectralData.attribute("spectral_energy_x"), Double.parseDouble(data[0]));
			iDataPoint.setValue(rawSpectralData.attribute("spectral_energy_y"), Double.parseDouble(data[1]));
			iDataPoint.setValue(rawSpectralData.attribute("spectral_energy_z"), Double.parseDouble(data[2]));
			iDataPoint.setValue(rawSpectralData.attribute("fourier_coeff_x"), Double.parseDouble(data[3]));
			iDataPoint.setValue(rawSpectralData.attribute("fourier_coeff_y"), Double.parseDouble(data[4]));
			iDataPoint.setValue(rawSpectralData.attribute("fourier_coeff_z"), Double.parseDouble(data[5]));
			iDataPoint.setValue(rawSpectralData.attribute("spectral_entropy_x"), Double.parseDouble(data[6]));
			iDataPoint.setValue(rawSpectralData.attribute("spectral_entropy_y"), Double.parseDouble(data[7]));
			iDataPoint.setValue(rawSpectralData.attribute("spectral_entropy_z"), Double.parseDouble(data[8]));
			rawSpectralData.add(iDataPoint);
		}
		reader.close();

		if (rawData.numInstances() != 0) {
			Instance iDataPoint = transformData(rawData, rawSpectralData, userid);
			m_Data.add(iDataPoint);
		}

	}

}
