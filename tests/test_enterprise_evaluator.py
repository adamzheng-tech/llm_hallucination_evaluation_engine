import argparse
import asyncio
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fitz
from aiohttp import web
from core.enterprise_evaluator import (Chunk, Retriever, extract_claims,
    ingest_document, run, validate_judgment)

ROOT = Path(__file__).resolve().parents[1]


class LocalTests(unittest.TestCase):
    def test_retrieval_preserves_evidence_and_handles_no_overlap(self):
        retriever = Retriever([Chunk('page:2', 'Memory is fixed at sixteen gigabytes.'),
                               Chunk('page:5', 'Warranty lasts two years.')])
        self.assertEqual(retriever.retrieve('memory', 1)[0]['id'], 'page:2')
        self.assertEqual(retriever.retrieve('zebra', 2), [])

    def test_pdf_empty_page_does_not_shift_page_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.pdf'
            with fitz.open() as document:
                document.new_page()
                document.new_page().insert_text((72, 72), 'Memory is fixed.')
                document.save(path)
            self.assertEqual(ingest_document(path)[0].id, 'page:2')

    def test_claim_contract(self):
        self.assertEqual(len(extract_claims(ROOT / 'examples/claims.md')), 3)

    def test_validation_rejects_fabricated_quote_and_missing_evidence(self):
        chunks = [{'id': 'page:1', 'text': 'Memory is fixed.'}]
        good = {'status': 'CONTRADICTED', 'reason': 'Cannot upgrade.',
                'evidence': [{'chunk_id': 'page:1', 'quote': 'Memory is fixed.'}]}
        self.assertEqual(validate_judgment(good, chunks), good)
        for evidence in [[], [{'chunk_id': 'page:1', 'quote': 'Memory is expandable.'}],
                         [{'chunk_id': 'page:9', 'quote': 'Memory is fixed.'}]]:
            with self.assertRaises(ValueError):
                validate_judgment({**good, 'evidence': evidence}, chunks)

    def test_validation_rejects_wrong_shape_and_status(self):
        for value in [[], {}, {'status': 'PASS', 'reason': 'ok', 'evidence': []}]:
            with self.assertRaises(ValueError):
                validate_judgment(value, [])


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.calls = 0
        self.mode = 'success'
        async def completion(request):
            self.calls += 1
            if self.mode == 'auth':
                return web.Response(status=401)
            if self.mode == 'retry' and self.calls == 1:
                return web.Response(status=503)
            body = await request.json()
            self.assertEqual(body['model'], 'local-test')
            data = json.loads(body['messages'][1]['content'])
            quote = data['passages'][0]['text']
            result = {'status': 'SUPPORTED', 'reason': 'Mock transport test, not accuracy.',
                      'evidence': [{'chunk_id': data['passages'][0]['id'], 'quote': quote}]}
            if self.mode == 'invalid':
                result['evidence'][0]['quote'] = 'invented quotation'
            return web.json_response({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]})
        app = web.Application()
        app.router.add_post('/v1/chat/completions', completion)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        import socket
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        self.port = sock.getsockname()[1]
        await web.SockSite(self.runner, sock).start()
        self.args = argparse.Namespace(source=ROOT / 'examples/reference.txt',
            claims=ROOT / 'examples/claims.md', output=Path(self.temp.name) / 'result.csv',
            dry_run=False, model='local-test', base_url=f'http://127.0.0.1:{self.port}/v1',
            concurrency=1, timeout=2, top_k=2, attempts=2, proxy=None, response_format='json_object')

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.temp.cleanup()

    async def execute(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only'}):
            return await run(self.args)

    async def test_transport_and_csv(self):
        self.assertEqual(await self.execute(), 0)
        with self.args.output.open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 3)
        self.assertTrue(json.loads(rows[0]['evidence']))

    async def test_dry_run_never_calls_provider(self):
        self.args.dry_run = True
        self.assertEqual(await self.execute(), 0)
        self.assertEqual(self.calls, 0)

    async def test_auth_not_retried(self):
        self.mode = 'auth'
        self.assertEqual(await self.execute(), 1)
        self.assertEqual(self.calls, 3)

    async def test_transient_error_retried(self):
        self.mode = 'retry'
        self.assertEqual(await self.execute(), 0)
        self.assertEqual(self.calls, 4)

    async def test_fabricated_evidence_fails(self):
        self.mode = 'invalid'
        self.assertEqual(await self.execute(), 1)


if __name__ == '__main__':
    unittest.main()
