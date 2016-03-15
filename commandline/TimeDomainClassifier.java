import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class TimeDomainClassifier extends DataClassifier {

	public TimeDomainClassifier(String[] classes) {
		super(classes);
	}

	protected Instances initializeTransformedDataFormat() {
		int numAttributes = 19;
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
		// Declare class attribute
		FastVector fvUserVal = new FastVector(classes.length);
		for (int i = 0; i < classes.length; i++) {
			fvUserVal.addElement(classes[i]);
		}
		Attribute classAttribute = new Attribute("userid", fvUserVal);
		fvTransformedWekaAttributes.addElement(classAttribute);
		Instances transformedData = new Instances(
				"FeaturesV2" /* relation name */,
				fvTransformedWekaAttributes /* attribute vector */,
				25 /* initial capacity */);
		transformedData.setClassIndex(transformedData.numAttributes() - 1);
		return transformedData;
	}

	protected Instance transformData(Instances rawData, String userid) {
		String[] axis = new String[] { "x", "y", "z" };
		Instance windowDataPoint = new DenseInstance(19);
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
		// Class attribute
		windowDataPoint.setValue(m_Data.attribute("userid"), userid);

		return windowDataPoint;
	}

}
