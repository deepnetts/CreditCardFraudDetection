package com.deepnetts.examples.creditcardfraud;

import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;
import tech.tablesaw.api.*;
import tech.tablesaw.columns.Column;
import tech.tablesaw.io.csv.CsvReadOptions;
import tech.tablesaw.plotly.Plot;

import tech.tablesaw.plotly.api.BoxPlot;
import tech.tablesaw.plotly.api.VerticalBarPlot;

import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.components.Marker;
import tech.tablesaw.plotly.traces.BarTrace;
import tech.tablesaw.plotly.traces.HistogramTrace;

/**
 * Data preparation utilities for dataframe from TableSaw.
 */
public class DataPreparation {

    private final Table dataSet;   
       
    public DataPreparation(Table dataSet) {
        this.dataSet = dataSet;
    }

    public Table getData() {
        return dataSet;
    }

    public DataPreparation(String csvFile) {
        CsvReadOptions options = CsvReadOptions.builder(csvFile)
                .header(true)
                .sample(false) // Read the entire file
                .missingValueIndicator("NaN", "", " ", "-", "nan", "NULL")
                .build();

        this.dataSet = Table.read().usingOptions(options);
    }

    public DataPreparation(CsvReadOptions options) {
        this.dataSet = Table.read().usingOptions(options);
    }

 
    public void countMissingValues() {
        int totalRows = dataSet.rowCount();

        System.out.println(">> Overview of missing values per column:");

        for (Column<?> column : dataSet.columns()) {
            int missing = column.countMissing();
            double percent = (missing * 100.0) / totalRows;

            System.out.printf("Column: %s | Missing: %d (%.2f%%)\n", column.name(), missing, percent);

            if (missing == 0) {
                continue;
            }

            // 25 < , use mean for normal dist, median for non nrmal
            
            if (percent >= 25 && percent <= 40) {
                System.out.println("   -> Recommendation: Consider using more sophisticated imputation methods (e.g., KNN, regression, model-based).");
            } else if (percent > 40) {
                System.out.println("   -> WARNING: Missing percentage is too high. Consider dropping the variable/column.");
                // @FIX:umesto System.out koristiti LOGGER svuda
            //    LOGGER.info("   -> WARNING: Missing percentage is too high. Consider dropping the variable/column.");
            }
        }
    }


    public Table handleMissingValues() {
        List<String> columnsToRemove = new ArrayList<>();
        int totalRows = dataSet.rowCount();

        for (Column<?> column : dataSet.columns()) {
            int missing = column.countMissing(); // izbeci da ovo radi ponovo ako je vec uradio gore sa brojanjem
            double percent = (missing * 100.0) / totalRows;

            System.out.printf("Column: %s | Missing: %d (%.2f%%)\n", column.name(), missing, percent);

            if (missing == 0) {
                continue;
            }

            if (percent > 40) { // best practcie, over 35
                System.out.printf(">> Column %s has too many missing values. It may be removed.\n", column.name());
                columnsToRemove.add(column.name());
                continue;
            }

            if (column instanceof StringColumn) {
                StringColumn sc = (StringColumn) column;
                Table freqTable = sc.countByCategory().sortDescendingOn("Count");
                String mostFrequent = freqTable.stringColumn("Category").get(0); 

                sc.set(sc.isMissing(), mostFrequent);
                System.out.printf(">> Categorical column '%s' - missing values replaced with the most frequent: %s\n", column.name(), mostFrequent);
            } else if (column instanceof DoubleColumn) {
                DoubleColumn dc = (DoubleColumn) column;
                DoubleColumn filtered = dc.where(dc.isNotMissing());

                double[] values = filtered.asDoubleArray();
                double mean = filtered.mean();
                double std = filtered.standardDeviation();

                KolmogorovSmirnovTest ksTest = new KolmogorovSmirnovTest();
                NormalDistribution normalDist = new NormalDistribution(mean, std);

                double pValue = ksTest.kolmogorovSmirnovTest(normalDist, values);
                boolean isNormal = pValue > 0.05;

                if (isNormal) {
                    dc.set(dc.isMissing(), mean);
                    System.out.printf(">> Column '%s' is normally distributed (p=%.4f), NaNs replaced with mean: %.2f\n", column.name(), pValue, mean);
                } else {
                    double median = filtered.median();
                    dc.set(dc.isMissing(), median);
                    System.out.printf(">> Column '%s' is not normally distributed (p=%.4f), NaNs replaced with median: %.2f\n", column.name(), pValue, median);
                }
            }
        }

        System.out.println("\nThe following columns may be removed due to a high percentage of missing values:");
        for (String colName : columnsToRemove) {
            System.out.println(" - " + colName);
        }

        return dataSet;
    }
   
    /**
     * Replaces missing values with most frequent value
     * @param columnName 
     */
    public void imputeCategorical(String columnName) {

        Column<?> column = dataSet.column(columnName);

        StringColumn asString = column.asStringColumn();

        Table freqTable = asString.countByCategory().sortDescendingOn("Count");
        String mostFrequent = freqTable.stringColumn("Category").get(0);

        for (int i = 0; i < column.size(); i++) {
            if (column.isMissing().contains(i)) {
                try {
                    if (column instanceof StringColumn) {
                        ((StringColumn) column).set(i, mostFrequent);
                    } else if (column instanceof IntColumn) {
                        int value = Integer.parseInt(mostFrequent);
                        ((IntColumn) column).set(i, value);
                    } else if (column instanceof DoubleColumn) { // ovo i nema smisla za kategorijske i ovo dole
                        double value = Double.parseDouble(mostFrequent);
                        ((DoubleColumn) column).set(i, value);
                    } else if (column instanceof LongColumn) {
                        long value = Long.parseLong(mostFrequent);
                        ((LongColumn) column).set(i, value);
                    } else {
                        System.out.printf(">> [WARNING] Column '%s' has unsupported type: %s\n", column.name(), column.type());
                    }
                } catch (NumberFormatException e) {
                    System.out.printf(">> [ERROR] Failed to parse '%s' for column '%s'\n", mostFrequent, column.name());
                }
            }
        }

        System.out.printf(">> Column '%s' (treated as categorical) - missing values replaced with the most frequent: %s\n",
                column.name(), mostFrequent);
    }

    public void imputeNumeric(String columnName) {

        DoubleColumn column = (DoubleColumn) dataSet.column(columnName);

        DoubleColumn filtered = column.where(column.isNotMissing());
        double[] values = filtered.asDoubleArray();
        double mean = filtered.mean();
        double std = filtered.standardDeviation();

        KolmogorovSmirnovTest ksTest = new KolmogorovSmirnovTest();
        NormalDistribution normalDist = new NormalDistribution(mean, std);
        double pValue = ksTest.kolmogorovSmirnovTest(normalDist, values);
        boolean isNormal = pValue > 0.05;

        if (isNormal) {
            column.set(column.isMissing(), mean);
            System.out.printf(">> Column '%s' is normally distributed (p=%.4f), NaNs replaced with mean: %.2f\n",
                    column.name(), pValue, mean);
        } else {
            double median = filtered.median();
            column.set(column.isMissing(), median);
            System.out.printf(">> Column '%s' is not normally distributed (p=%.4f), NaNs replaced with median: %.2f\n",
                    column.name(), pValue, median);
        }
    }

    public void showBoxPlotsAndOutliers(List<String> columnNames) {
        for (String columnName : columnNames) {
            Column<?> col = dataSet.column(columnName);

            if (col instanceof NumericColumn<?>) {
                NumericColumn<?> numCol = (NumericColumn<?>) col;
                String numericColName = numCol.name();

                double q1 = numCol.quartile1();
                double q3 = numCol.quartile3();
                double iqr = q3 - q1;
                double lowerBound = q1 - 1.5 * iqr;
                double upperBound = q3 + 1.5 * iqr;

                NumericColumn<?> outliers = numCol.where(numCol.isLessThan(lowerBound).or(numCol.isGreaterThan(upperBound)));

                System.out.println("Column: " + numericColName);
                System.out.println("   Total outliers: " + outliers.size());
                if (outliers.size() > 0) {
                    System.out.println("   Outlier values: " + outliers.unique().asList());
                } else {
                    System.out.println("   No outliers detected.");
                }

                // Dummy kolona za boxplot
                String dummyGroupColName = "group_for_" + numericColName;
                StringColumn groupCol = StringColumn.create(dummyGroupColName, dataSet.rowCount());
                groupCol.setMissingTo("All");
                dataSet.addColumns(groupCol);

                // Kreiranje i prikazivanje boxplot-a
                Figure fig = BoxPlot.create("Boxplot - " + numericColName, dataSet, dummyGroupColName, numericColName);
                Plot.show(fig);

                // Uklanjanje dummy kolone
                dataSet.removeColumns(dummyGroupColName);
                System.out.println("--------------------------------------------------");
            } else {
                System.out.println("Column '" + columnName + "' is not numeric and will not be plotted.");
            }
        }
    }

    public void countUniqueValues() {
        for (Column<?> col : dataSet.columns()) {
            String colName = col.name();

            int uniqueCount = col.unique().size();
            System.out.println("Column: " + colName + " has " + uniqueCount + " unique values.");
        }
    }

    public void winsorize(String columnName) {
        Column<?> col = dataSet.column(columnName);

        if (col instanceof NumericColumn<?>) {
            NumericColumn<?> numCol = (NumericColumn<?>) col;

            if (numCol instanceof DoubleColumn) {
                DoubleColumn doubleCol = (DoubleColumn) numCol;

                double q1 = doubleCol.quartile1();
                double q3 = doubleCol.quartile3();
                double iqr = q3 - q1;

                double lowerBound = q1 - 1.5 * iqr;
                double upperBound = q3 + 1.5 * iqr;

                if (lowerBound < 0) {
                    lowerBound = 0;
                }

                for (int i = 0; i < doubleCol.size(); i++) {
                    double value = doubleCol.get(i);
                    if (value < lowerBound) {
                        doubleCol.set(i, lowerBound);
                    } else if (value > upperBound) {
                        doubleCol.set(i, upperBound);
                    }
                }

                System.out.println("Winsorization completed for column: " + columnName);
                System.out.println("Lower bound: " + lowerBound);
                System.out.println("Upper bound: " + upperBound);

            } else {
                System.out.println("Column '" + columnName + "' is not a DoubleColumn and cannot be winsorized.");
            }
        } else {
            System.out.println("Column '" + columnName + "' is not numeric and cannot be winsorized.");
        }
    }

    public void previewRows(int numberOfRows) {
        System.out.println("Loaded dataset (first "+numberOfRows+" rows):");
        System.out.println(dataSet.first(numberOfRows));
    }

    public void statistics() {
        System.out.println(dataSet.summary());
    }

    public void columnInfo() {
        System.out.println(dataSet.structure());
    }

    public void plotCategoricalFeatureVsClassTarget(String featureCol, String targetCol) {

        Table grouped = dataSet.countBy(featureCol, targetCol);
        String countCol = "Count";

        StringColumn featureStr = grouped.column(featureCol).asStringColumn();
        StringColumn targetStr = grouped.column(targetCol).asStringColumn();

        List<String> featureValues = dataSet.column(featureCol).asStringColumn().unique().asList();
        List<String> targetValues = dataSet.column(targetCol).asStringColumn().unique().asList();

        List<BarTrace> traces = new ArrayList<>();

        for (String targetVal : targetValues) {
            List<String> xValues = new ArrayList<>();
            List<Number> yValues = new ArrayList<>();

            for (String featureVal : featureValues) {

                Table filtered = grouped.where(
                        featureStr.isEqualTo(featureVal).and(targetStr.isEqualTo(targetVal))
                );

                int count = filtered.isEmpty() ? 0 : filtered.intColumn(countCol).get(0);
                xValues.add(featureVal);
                yValues.add(count);
            }

            // Kreiranje trace-a
            BarTrace trace = BarTrace.builder(
                    StringColumn.create("x", xValues),
                    DoubleColumn.create("y", yValues)
            ).name(targetVal).build();

            traces.add(trace);
        }

        Layout layout = Layout.builder()
                .title("Plot " + targetCol + " againts " + featureCol)
                .build();

        Plot.show(new Figure(layout, traces.toArray(new BarTrace[0])));
    }

    public void plotNumericFeatureVsClassTarget(String featureCol, String targetCol) {

        IntColumn target = dataSet.intColumn(targetCol);
        DoubleColumn feature = dataSet.doubleColumn(featureCol);

        String[] colors = {"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"};

        List<HistogramTrace> traces = new ArrayList<>();

        for (int targetClass : target.unique()) {
            Table filtered = dataSet.where(target.isEqualTo(targetClass));

            HistogramTrace trace = HistogramTrace.builder(filtered.doubleColumn(featureCol))
                    .name("Class " + targetClass)
                    .marker(Marker.builder().color(colors[targetClass % colors.length]).opacity(0.6).build())
                    .build();
            traces.add(trace);
        }

        Layout layout = Layout.builder()
                .title("Distribution of " + featureCol + " by class " + targetCol)
                .build();

        Figure fig = new Figure(layout, traces.toArray(new HistogramTrace[0]));

        Plot.show(fig);
    }
    
 public void checkClassBalance(String columnName) { 
     checkClassBalance(dataSet, columnName);
 }   
    
 public static void checkClassBalance(Table dataSet, String columnName) { 
        Table negativeExamples = dataSet.where(t -> t.intColumn(columnName).isEqualTo(0));
        Table positiveExamples = dataSet.where(t -> t.intColumn(columnName).isEqualTo(1));

        int numPositive = positiveExamples.rowCount();
        int numNegative = dataSet.rowCount() - numPositive;
        
        String[] categories = {"Not Fraud", "Fraud"};
        int[] values = {numNegative, numPositive};
        
        Table barTable = Table.create("Dist bar")
                .addColumns(
                        StringColumn.create("Categories", categories),
                        IntColumn.create("Values", values)
                );
        Plot.show(VerticalBarPlot.create("Distribution of Fraud and Not Fraud Examples", barTable, "Categories", "Values"));

        System.out.println("Positive class rows: "+numPositive);
        System.out.println("Negative class rows: "+numNegative);   
    }
       
    public static TabularDataSet<MLDataItem> createDataSet(Table table) {
        TabularDataSet<MLDataItem> dataSet= new TabularDataSet(29, 1);
        String[] colNAmes = new String[table.columns().size()];
        table.columnNames().toArray(colNAmes);
        dataSet.setColumnNames(colNAmes);
        
        for (int i = 0; i < table.rowCount(); i++) {            
            Row row = table.row(i);
            
            float[] in = new float[29];
            for(int c=0; c<29; c++) {
                in[c] = (float)row.getDouble(c);
            }
            
            float[] targetOut = new float[1];
            targetOut[0] = row.getInt(29);
                        
            MLDataItem tableRow = new TabularDataSet.Item(in, targetOut);
            dataSet.add(tableRow);
        }
        return dataSet;
    }
    
    public static Table createBalancedSample(Table dataTable, String columnName, long rndSeed) {
        Table negative = dataTable.where(t -> t.intColumn(columnName).isEqualTo(0));
        Table positive = dataTable.where(t -> t.intColumn(columnName).isEqualTo(1));
        
        Table balancedSample = Table.create("Balanced CCF", positive.columns());
        balancedSample.append(positive); // get all positive examples
               
        Random random = new Random(rndSeed);
        IntStream rndIdxs = random.ints(positive.rowCount(), 0, negative.rowCount());              
        int[] randomIdx = rndIdxs.toArray();        
                 
        balancedSample.append(negative.rows(randomIdx));
        return balancedSample;
    }    


}
