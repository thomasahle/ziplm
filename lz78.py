
class LZ78:
    def __init__(self):
        pass

    @staticmethod
    def compress(data):
        """ Compress the data using the LZ78 algorithm.

        :param data: A string containing the data to be compressed.
        :return: A list of (index, char) pairs representing the compressed data.
        """
        dictionary = {"": 0}
        w = ""
        result = []

        for c in data:
            wc = w + c
            # Keep adding to w as long as we are still in the dictionary
            if wc in dictionary:
                w = wc
            # Once we are no longer in the dictionary, we know the previous
            # token was
            else:
                yield (dictionary[w], c)
                dictionary[wc] = len(dictionary)
                w = ""

        # Check if there's remaining data to be compressed
        yield (dictionary[w], "")

    @staticmethod
    def decompress(compressed_data):
        """ Decompress the data compressed using the LZ78 algorithm.

        :param compressed_data: A list of (index, char) pairs representing the compressed data.
        :return: A string containing the decompressed data.
        """
        dictionary = [""]

        result = []

        for index, char in compressed_data:
            entry = dictionary[index] + char
            result.append(entry)
            dictionary.append(entry)

        return ''.join(result)


# Testing the LZ78 implementation
if __name__ == '__main__':
    data = "ABABABABA"
    print(f"Original Data: {data}")

    compressed = list(LZ78.compress(data))
    print(f"Compressed Data: {compressed}")

    decompressed = LZ78.decompress(compressed)
    print(f"Decompressed Data: {decompressed}")
