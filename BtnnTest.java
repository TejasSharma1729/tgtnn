import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

class BTNN {
    double[][] data;
    double[][][] index;
    int N;
    int D;
    static int L = 10;

    public BTNN(double[][] data) {
        this.data = data;
        this.N = data.length;
        this.D = data[0].length;
        for (int i = 1; i < N; i++) {
            if (data[i].length != D) {
                throw new IllegalArgumentException(
                    "All data points must have the same dimension."
                );
            }
        }
    }

    static double dotProduct(double[] a, double[] b) {
        double result = 0;
        for (int i = 0; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    class Neighbor implements Comparable<Neighbor> {
        public int base;
        public int offset;
        public double query_dot;

        public Neighbor(int base, int offset, double query_dot) {
            this.base = base;
            this.offset = offset;
            this.query_dot = query_dot;
        }

        @Override
        public int compareTo(Neighbor other) {
            int cmp = Double.compare(this.query_dot, other.query_dot);
            if (cmp != 0) {
                return cmp;
            }
            int thisIndex = this.base * (1 << BTNN.L) + this.offset;
            int otherIndex = other.base * (1 << BTNN.L) + other.offset;
            return Integer.compare(thisIndex, otherIndex);
        }
    }

    public int numFeatures() { return this.N; }
    public int dimension() { return this.D; }

    public void buildIndex() {
        const int G = 1 << BTNN.L; // Number of grid cells per dimension
        const int Ng = (this.N + G - 1) / G; // Number of grid cells needed to cover all points
        this.index = new double[Ng][2 * G][D];

        for (int bIndex = 0; bIndex < this.N; bIndex += G) {
            double[][] indexTree = this.index[bIndex / G];
            for (int offset = 0; offset < G; offset++) {
                if (bIndex + offset >= this.N) {
                    break;
                }
                double[] point = this.data[bIndex + offset];
                for (int d = 0; d < this.D; d++) {
                    indexTree[offset + G][d] = point[d];
                }
            }
            for (int offset = G - 1; offset > 0; offset--) {
                for (int d = 0; d < this.D; d++) {
                    indexTree[offset][d] = indexTree[2 * offset][d] + indexTree[2 * offset + 1][d];
                }
            }
            this.index[bIndex / G] = indexTree; // Should not mater.
        }
    }

    public int[] query(double[] point, int K) {
        int[] neighbors = new int[K];
        int numNeighbors = 0;
        const int G = 1 << BTNN.L;
        const int Ng = (this.N + G - 1) / G;

        PriorityQueue<Neighbor> pq = new PriorityQueue<>(Collections.reverseOrder());
        for (int bIndex = 0; bIndex < Ng; bIndex++) {
            double[] root = this.index[bIndex][1];
            double root_dot = BTNN.dotProduct(root, point);
            pq.offer(new Neighbor(bIndex, 1, root_dot));
        }
        
        while (!pq.isEmpty() && numNeighbors < K) {
            Neighbor neighbor = pq.poll();
            int base = neighbor.base;
            int offset = neighbor.offset;
            if (offset < G) {
                int leftOffset = offset * 2;
                int rightOffset = offset * 2 + 1;
                
                double[] leftNode = this.index[base][leftOffset];
                double leftDot = BTNN.dotProduct(leftNode, point);
                pq.offer(new Neighbor(base, leftOffset, leftDot));

                double[] rightNode = this.index[base][rightOffset];
                double rightDot = BTNN.dotProduct(rightNode, point);
                pq.offer(new Neighbor(base, rightOffset, rightDot));
            }
            else {
                int dataIndex = base * G + offset - G;
                if (dataIndex < this.N) {
                    neighbors[numNeighbors++] = dataIndex;
                }
            }
        }

        return neighbors;
    }

    public int[] naiveSearch(double[] point, int K) {
        PriorityQueue<Neighbor> pq = new PriorityQueue<>(Collections.reverseOrder());
        for (int i = 0; i < this.N; i++) {
            double dot = BTNN.dotProduct(this.data[i], point);
            pq.offer(new Neighbor(0, i, dot));
        }
        int[] neighbors = new int[K];
        for (int i = 0; i < K; i++) {
            neighbors[i] = pq.poll().offset;
        }
        return neighbors;
    }
}


class NpyReader {
    static double[][] read(String fileName) {
        try (InputStream inputStream = new BufferedInputStream(Files.newInputStream(Paths.get(fileName)))) {
            DataInputStream dataInputStream = new DataInputStream(inputStream);

            byte[] magic = new byte[6];
            dataInputStream.readFully(magic);
            if (magic[0] != (byte) 0x93 || magic[1] != 'N' || magic[2] != 'U' || magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
                throw new IllegalArgumentException("Not a NumPy .npy file: " + fileName);
            }

            int majorVersion = dataInputStream.readUnsignedByte();
            int minorVersion = dataInputStream.readUnsignedByte();

            int headerLength;
            if (majorVersion == 1) {
                headerLength = readLittleEndianUnsignedShort(dataInputStream);
            } else if (majorVersion == 2 || majorVersion == 3) {
                headerLength = readLittleEndianInt(dataInputStream);
            } else {
                throw new IllegalArgumentException("Unsupported .npy version: " + majorVersion + "." + minorVersion);
            }

            byte[] headerBytes = new byte[headerLength];
            dataInputStream.readFully(headerBytes);
            String header = new String(headerBytes, StandardCharsets.ISO_8859_1).trim();

            String descr = extractHeaderValue(header, "descr");
            if (!"<f8".equals(descr) && !"=f8".equals(descr) && !"|f8".equals(descr)) {
                throw new IllegalArgumentException("Expected float64 .npy data, got descr=" + descr);
            }

            String fortranOrder = extractHeaderValue(header, "fortran_order");
            if (!"False".equals(fortranOrder)) {
                throw new IllegalArgumentException("Fortran-ordered arrays are not supported: " + fileName);
            }

            long[] shape = parseShape(extractHeaderValue(header, "shape"));
            if (shape.length != 2) {
                throw new IllegalArgumentException("Expected a 2D array in " + fileName + ", got " + Arrays.toString(shape));
            }

            int rows = Math.toIntExact(shape[0]);
            int cols = Math.toIntExact(shape[1]);
            byte[] dataBytes = dataInputStream.readAllBytes();
            ByteBuffer buffer = ByteBuffer.wrap(dataBytes).order(ByteOrder.LITTLE_ENDIAN);

            double[][] values = new double[rows][cols];
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    values[row][col] = buffer.getDouble();
                }
            }

            if (buffer.hasRemaining()) {
                throw new IllegalArgumentException("Unexpected extra bytes in " + fileName);
            }

            return values;
        } catch (IOException exception) {
            throw new RuntimeException("Failed to read NumPy file: " + fileName, exception);
        }
    }

    private static int readLittleEndianUnsignedShort(DataInputStream inputStream) throws IOException {
        int first = inputStream.readUnsignedByte();
        int second = inputStream.readUnsignedByte();
        return first | (second << 8);
    }

    private static int readLittleEndianInt(DataInputStream inputStream) throws IOException {
        int b0 = inputStream.readUnsignedByte();
        int b1 = inputStream.readUnsignedByte();
        int b2 = inputStream.readUnsignedByte();
        int b3 = inputStream.readUnsignedByte();
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }

    private static String extractHeaderValue(String header, String key) {
        String prefix = "'" + key + "':";
        int keyIndex = header.indexOf(prefix);
        if (keyIndex < 0) {
            throw new IllegalArgumentException("Missing key in .npy header: " + key);
        }

        int valueStart = keyIndex + prefix.length();
        while (valueStart < header.length() && Character.isWhitespace(header.charAt(valueStart))) {
            valueStart++;
        }

        if (valueStart >= header.length()) {
            throw new IllegalArgumentException("Malformed .npy header: " + header);
        }

        char first = header.charAt(valueStart);
        if (first == '\'') {
            int end = header.indexOf('\'', valueStart + 1);
            if (end < 0) {
                throw new IllegalArgumentException("Malformed .npy header: " + header);
            }
            return header.substring(valueStart + 1, end);
        }

        if (first == '(') {
            int end = header.indexOf(')', valueStart);
            if (end < 0) {
                throw new IllegalArgumentException("Malformed .npy header: " + header);
            }
            return header.substring(valueStart, end + 1);
        }

        int end = valueStart;
        while (end < header.length() && header.charAt(end) != ',' && header.charAt(end) != '}') {
            end++;
        }
        return header.substring(valueStart, end).trim();
    }

    private static long[] parseShape(String shapeText) {
        String trimmed = shapeText.trim();
        if (!trimmed.startsWith("(") || !trimmed.endsWith(")")) {
            throw new IllegalArgumentException("Malformed .npy shape: " + shapeText);
        }

        String body = trimmed.substring(1, trimmed.length() - 1).trim();
        if (body.isEmpty()) {
            return new long[0];
        }

        String[] parts = body.split(",");
        List<Long> dimensions = new ArrayList<>();
        for (String part : parts) {
            String dimension = part.trim();
            if (dimension.isEmpty()) {
                continue;
            }
            dimensions.add(Long.parseLong(dimension));
        }

        long[] shape = new long[dimensions.size()];
        for (int i = 0; i < dimensions.size(); i++) {
            shape[i] = dimensions.get(i);
        }
        return shape;
    }
}


public class BtnnTest {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java BtnnTest <dataset> [num_queries]");
            System.exit(1);
        }

        String dataFile = "data/" + args[0] + "/X.npy";
        String queryFile = "data/" + args[0] + "/Q.npy";
        double[][] data = NpyReader.read(dataFile);
        double[][] queries = NpyReader.read(queryFile);

        int Nq = queries.length;
        if (args.length > 1) {
            Nq = Integer.parseInt(args[1]);
        }

        BTNN btnn = new BTNN(data);
        btnn.buildIndex();

        double meanTime = 0.0;
        double meanNaiveTime = 0.0;
        double meanRecall = 0.0;

        for (double[] query : queries) {
            long startTime = System.nanoTime();
            int[] neighbors = btnn.query(query, 10);
            long midTime = System.nanoTime();
            int[] naiveNeighbors = btnn.naiveSearch(query, 10);
            long endTime = System.nanoTime();

            double precision = 0.0;
            for (int neighbor : neighbors) {
                for (int naiveNeighbor : naiveNeighbors) {
                    if (neighbor == naiveNeighbor) {
                        precision += 1.0 / 10.0;
                        break;
                    }
                }
            }
            meanTime += (midTime - startTime) / 1e9;
            meanNaiveTime += (endTime - midTime) / 1e9;
            meanRecall += precision;
        }

        meanTime /= Nq;
        meanNaiveTime /= Nq;
        meanRecall /= Nq;

        System.out.printf("Average query time: %.6f seconds%n", meanTime);
        System.out.printf("Average naive query time: %.6f seconds%n", meanNaiveTime);
        System.out.printf("Average recall: %.2f%%%n", meanRecall * 100);
    }
} 