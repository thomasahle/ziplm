import io, gzip, zlib

class IncrementalGzipCompressor:
    def __init__(self):
        self.buffer = io.BytesIO()
        # self.buffer = io.StringIO()
        self.compressor = gzip.GzipFile(fileobj=self.buffer, mode='wb')

    def add_and_get_difference(self, char):
        self.compressor.flush()
        start_size = self.buffer.tell()  # get current size
        self.compressor.write(char.encode())
        self.compressor.flush()  # flush to make sure all data is written
        end_size = self.buffer.tell()
        return end_size - start_size

    def close(self):
        self.compressor.close()


def compressed(s: str) -> int:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as f:
        f.write(s.encode())
        print(buf.tell(), f.tell())
        f.flush()
        #f.flush(zlib.Z_NO_FLUSH)
        #f.flush(zlib.Z_PARTIAL_FLUSH)
        #f.flush(zlib.Z_SYNC_FLUSH)
        #f.flush(zlib.Z_FULL_FLUSH)
        #print(f'{f.tell()=}', f.unused_data)
        print(dir(f))
    print(f'{buf.tell()=}')
    return buf.getvalue()


# Testing
base_string = "ABCABCABC"
chars = "ABCD"

compressor = IncrementalGzipCompressor()
part_a = compressor.add_and_get_difference(base_string)
part_b = compressor.add_and_get_difference(chars)
print(part_a, part_b, part_a + part_b)
print()

print(len(compressor.buffer.getvalue()))
compressor.close()
print(len(compressor.buffer.getvalue()))
print()

compressor2 = IncrementalGzipCompressor()
print(compressor2.add_and_get_difference(base_string + chars))
print(compressor2.buffer.tell())
compressor2.close()
print(compressor2.buffer.tell())
print()

compressor2 = IncrementalGzipCompressor()
print(compressor2.add_and_get_difference(base_string))
print(compressor2.buffer.tell())
compressor2.close()
print(compressor2.buffer.tell())
print()

print(len(gzip.compress((base_string).encode())))
print(len(gzip.compress((base_string + chars).encode())))

print(len(gzip.compress(base_string.encode())), len(compressed(base_string)))
print(gzip.decompress(compressed(base_string)))

compressor.close()

