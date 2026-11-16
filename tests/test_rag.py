import unittest

from rag import chunk_text, cosine_similarity, extract_pages, retrieve_chunks


class ChunkTextTests(unittest.TestCase):
    def test_normalizes_whitespace_and_keeps_chunks_overlapping(self):
        chunks = chunk_text("alpha beta gamma delta epsilon", chunk_size=16, overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 16 for chunk in chunks))
        self.assertEqual(" ".join(chunks).replace("  ", " "), "alpha beta gamma delta epsilon")
        self.assertTrue(chunks[0].endswith(chunks[1].split()[0]))

    def test_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=10, overlap=10)

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text(" \n\t "), [])


class RetrievalTests(unittest.TestCase):
    def test_cosine_similarity_handles_zero_vectors(self):
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_retrieves_most_similar_chunk_first(self):
        chunks = [
            {"text": "unrelated", "source": "a.txt", "page": None},
            {"text": "relevant", "source": "b.pdf", "page": 2},
        ]
        matches = retrieve_chunks(
            [1.0, 0.0],
            chunks,
            [[0.0, 1.0], [1.0, 0.0]],
            top_k=1,
        )

        self.assertEqual(matches[0]["text"], "relevant")
        self.assertEqual(matches[0]["page"], 2)

    def test_rejects_mismatched_chunk_embeddings(self):
        with self.assertRaises(ValueError):
            retrieve_chunks([1.0], [{"text": "x"}], [])


class ExtractionTests(unittest.TestCase):
    def test_extracts_utf8_text_and_preserves_source(self):
        pages = extract_pages("notes.txt", b"\xef\xbb\xbfHello document")

        self.assertEqual(
            pages,
            [{"text": "Hello document", "source": "notes.txt", "page": None}],
        )

    def test_rejects_unsupported_file_type(self):
        with self.assertRaises(ValueError):
            extract_pages("notes.docx", b"")


if __name__ == "__main__":
    unittest.main()
