import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class GaitClassifier extends DataClassifier {

	public GaitClassifier(String[] classes, int classifierOption) {
		super(classes, classifierOption);
	}

	protected Instances initializeTransformedDataFormat() {
		int numAttributes = 44;
		String[] axis = new String[] { "x", "y", "z" };
		String[] classes = m_Classes;

		// Transformed Data
		FastVector fvTransformedWekaAttributes = new FastVector(numAttributes);

		// Numeric attributes

		// Average
		for (int i = 0; i < axis.length; i++) {
			Attribute attribute = new Attribute("Average " + axis[i]);
			fvTransformedWekaAttributes.addElement(attribute);
		}

		// Standard Deviation
		for (int i = 0; i < axis.length; i++) {
			Attribute attribute = new Attribute("Standard Deviation " + axis[i]);
			fvTransformedWekaAttributes.addElement(attribute);
		}

		// Average Absolute Difference
		for (int i = 0; i < axis.length; i++) {
			Attribute attribute = new Attribute("Average Absolute Difference " + axis[i]);
			fvTransformedWekaAttributes.addElement(attribute);
		}

		// Time between peaks
		for (int i = 0; i < axis.length; i++) {
			Attribute attribute = new Attribute("Time between peaks " + axis[i]);
			fvTransformedWekaAttributes.addElement(attribute);
		}

		// Binned Distribution
		for (int i = 0; i < 10 * axis.length; i++) {
			Attribute attribute = new Attribute("Binned Distribution " + axis[i / 10] + " " + Integer.toString(i % 10));
			fvTransformedWekaAttributes.addElement(attribute);
		}

		// Average Resultant Acceleration
		Attribute attribute = new Attribute("Average Resultant Acceleration");
		fvTransformedWekaAttributes.addElement(attribute);

		// Declare class attribute
		FastVector fvUserVal = new FastVector(classes.length);
		for (int i = 0; i < classes.length; i++) {
			fvUserVal.addElement(classes[i]);
		}
		Attribute classAttribute = new Attribute("userid", fvUserVal);
		fvTransformedWekaAttributes.addElement(classAttribute);
		Instances transformedData = new Instances("gait" /* relation name */,
				fvTransformedWekaAttributes /* attribute vector */,
				25 /* initial capacity */);
		transformedData.setClassIndex(transformedData.numAttributes() - 1);
		return transformedData;
	}

	protected Instance transformData(Instances rawData, String userid) {
		String[] axis = new String[] { "x", "y", "z" };
		double[] mean = new double[axis.length];
		Instance windowDataPoint = new DenseInstance(44); // 43 features + 1 for
															// class
		// Average and Standard Deviation
		for (int i = 0; i < axis.length; i++) {
			mean[i] = rawData.meanOrMode(i);
			double sd = Math.sqrt(rawData.variance(i));
			windowDataPoint.setValue(m_Data.attribute("Average " + axis[i]), mean[i]);
			windowDataPoint.setValue(m_Data.attribute("Standard Deviation " + axis[i]), sd);
		}
		// Average Absolute Difference
		double[] avgAbsDiff = new double[axis.length];
		for (int instIdx = 0; instIdx < rawData.numInstances(); instIdx++) {
			Instance currInst = rawData.instance(instIdx);
			for (int i = 0; i < axis.length; i++) {
				avgAbsDiff[i] += Math.abs(currInst.value(rawData.attribute(axis[i])) - mean[i]);
			}
		}
		for (int i = 0; i < axis.length; i++) {
			avgAbsDiff[i] /= rawData.numInstances();
			windowDataPoint.setValue(m_Data.attribute("Average Absolute Difference " + axis[i]), avgAbsDiff[i]);
		}
		// Time Between Peaks
		boolean[][] peaks = new boolean[rawData.numAttributes()][rawData.numInstances()];
		int dt = 1; // time delay of detection
		int m = 1; // peak threshold detection
		for (int instIdx = 0; instIdx < rawData.numInstances(); instIdx++) {
			int lowerBound = instIdx - dt;
			int upperBound = instIdx + dt;
			if (lowerBound < 0)
				lowerBound = 0;
			if (upperBound >= rawData.numInstances())
				upperBound = rawData.numInstances() - 1;
			Instance lowerInst = rawData.instance(lowerBound);
			Instance currInst = rawData.instance(instIdx);
			Instance upperInst = rawData.instance(upperBound);
			for (int i = 0; i < axis.length; i++) {
				if (currInst.value(i) - lowerInst.value(i) > m && currInst.value(i) - upperInst.value(i) > m) {
					peaks[i][instIdx] = true;
				}
			}
		}
		for (int i = 0; i < axis.length; i++) {
			double sumTimeBetwnPeaks = 0;
			double timeBetwnPeaks = 0;
			double numPeaks = 0;
			for (int j = 0; j < peaks[i].length; j++) {
				if (peaks[i][j] == true) {
					if (numPeaks != 0) {
						sumTimeBetwnPeaks += timeBetwnPeaks;
					}
					timeBetwnPeaks = 0;
					numPeaks++;
				} else {
					timeBetwnPeaks++;
				}
			}
			windowDataPoint.setValue(m_Data.attribute("Time between peaks " + axis[i]), sumTimeBetwnPeaks / numPeaks); // TBD
		}
		// Binned Distribution
		double[] binsX = new double[10];
		double[] binsY = new double[10];
		double[] binsZ = new double[10];
		double[] rangesX = new double[10];
		double[] rangesY = new double[10];
		double[] rangesZ = new double[10];
		double minimumX = rawData.attributeStats(0).numericStats.min;
		double maximumX = rawData.attributeStats(0).numericStats.max;
		double minimumY = rawData.attributeStats(1).numericStats.min;
		double maximumY = rawData.attributeStats(1).numericStats.max;
		double minimumZ = rawData.attributeStats(2).numericStats.min;
		double maximumZ = rawData.attributeStats(2).numericStats.max;
		double stepX = (maximumX - minimumX) / 10;
		double stepY = (maximumY - minimumY) / 10;
		double stepZ = (maximumZ - minimumZ) / 10;
		for (int i = 0; i < 10; i++) {
			rangesX[i] = minimumX + stepX * (i + 1);
			rangesY[i] = minimumY + stepY * (i + 1);
			rangesZ[i] = minimumZ + stepZ * (i + 1);
		}
		for (int instIdx = 0; instIdx < rawData.numInstances(); instIdx++) {
			Instance currInst = rawData.instance(instIdx);
			for (int i = 0; i < 10; i++) {
				if (currInst.value(0) <= rangesX[i]) {
					binsX[i] += 1;
					break;
				}
			}
			for (int i = 0; i < 10; i++) {
				if (currInst.value(1) <= rangesY[i]) {
					binsY[i] += 1;
					break;
				}
			}
			for (int i = 0; i < 10; i++) {
				if (currInst.value(2) <= rangesZ[i]) {
					binsZ[i] += 1;
					break;
				}
			}
		}
		for (int i = 0; i < 10; i++) {
			binsX[i] /= rawData.numInstances();
			binsY[i] /= rawData.numInstances();
			binsZ[i] /= rawData.numInstances();
		}
		for (int i = 0; i < 10 * axis.length; i++) {
			int dim = i / 10;
			int binNum = i % 10;
			double[] dimBins = null;
			switch (dim) {
			case 0:
				dimBins = binsX;
				break;
			case 1:
				dimBins = binsY;
				break;
			case 2:
				dimBins = binsZ;
				break;
			default:
				System.out.println("[transformData(Instances rawData)] Possible Error");
			}
			windowDataPoint.setValue(
					m_Data.attribute("Binned Distribution " + axis[dim] + " " + Integer.toString(binNum)),
					dimBins[binNum]);
		}
		// Average Resultant Acceleration
		double avgResultAccel = 0;
		for (int instIdx = 0; instIdx < rawData.numInstances(); instIdx++) {
			Instance currInst = rawData.instance(instIdx);
			avgResultAccel += Math.sqrt(
					Math.pow(currInst.value(0), 2) + Math.pow(currInst.value(1), 2) + Math.pow(currInst.value(2), 2));
		}
		avgResultAccel /= rawData.numInstances();
		windowDataPoint.setValue(m_Data.attribute("Average Resultant Acceleration"), avgResultAccel);
		// Class attribute
		windowDataPoint.setValue(m_Data.attribute("userid"), userid);

		return windowDataPoint;
	}

}
