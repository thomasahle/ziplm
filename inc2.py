
import zlib

class ProgressiveCompressor:
    def __init__(self):
        self.compressor = zlib.compressobj()
        self.compressed_data = bytearray()

    def add_data_and_compress(self, data: str) -> int:
        # Add data to compressor.
        # Some input may be kept in internal buffers for later processing.
        self.compressed_data += self.compressor.compress(data.encode())
        print(dir(self.compressor))
        return len(self.compressed_data)

    def finish(self):
        # Finalize the compression
        self.compressed_data += self.compressor.flush()

def main():
    compressor = ProgressiveCompressor()
    base_str = 'ABCDE'  # You can change this to any character or sequence you prefer
    total_length = 0

    for _ in range(10):  # Adjust the range as needed
        total_length += len(base_str)
        compressed_len = compressor.add_data_and_compress(base_str)
        print(f"Original Length: {total_length}, Compressed Length: {compressed_len}")

    # This step ensures any remaining data in the compressor is flushed and added to the result.
    compressor.finish()

if __name__ == "__main__":
    main()
