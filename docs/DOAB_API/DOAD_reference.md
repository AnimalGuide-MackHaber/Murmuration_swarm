Directory of Open Access Books (DOAB) API Reference

1. Overview

DOAB is a community-driven discovery service that indexes peer-reviewed open access books. It operates on a DSpace repository architecture. Because it aggregates data from multiple publishers, metadata quality and file formats vary significantly across the repository.

2. API Endpoints

The primary interface for programmatic extraction is the REST API.

Base URL: https://directory.doabooks.org/rest

Search Endpoint: /search

The expand Parameter Constraint

By default, the DOAB search endpoint returns only top-level item metadata (handle, name, ID). It drops the actual file links to save bandwidth. To extract PDF URLs, you must pass the expand=bitstreams parameter in your HTTP request.

Example Request:
https://directory.doabooks.org/rest/search?query=dc.subject.classification:"Medicine"&expand=bitstreams

3. Query Syntax

DOAB uses standard Lucene query syntax.

Full Text: query=cancer+AND+treatment

Title Search: query=dc.title:"clinical+anatomy"

Subject Search: query=dc.subject.classification:"Medicine"

4. Payload Structure & Extraction Logic

The API returns a JSON array of item objects. The file retrieval links are nested inside the bitstreams array of each item.

[
  {
    "id": 12345,
    "name": "Title of the Book",
    "handle": "20.500.12854/xxxxx",
    "bitstreams": [
      {
        "name": "book_file.pdf",
        "mimeType": "application/pdf",
        "retrieveLink": "/rest/bitstreams/abcde-12345/retrieve"
      }
    ]
  }
]


Extraction Protocol:

Check if bitstreams exists in the item.

Iterate through bitstreams.

Validate that mimeType equals application/pdf. (Do not skip this; DOAB hosts EPUBs, HTML files, and JPG covers in the same array).

Prepend the base domain (https://directory.doabooks.org) to the retrieveLink.

5. Inefficiencies and Operational Risks

Metadata Bleed: The dc.subject.classification is broad. A query for "Medicine" will return clinical texts alongside sociology, ethics, and history texts remotely related to healthcare. Client-side filtering of titles or abstracts is required for precision.

Rate Limiting: DOAB infrastructure cannot handle rapid concurrent requests. Implementing a static delay (e.g., 2 to 5 seconds) between API queries and file downloads is required to prevent IP blocking.

Incomplete Records: Some records are metadata-only or link to external publisher sites. If the bitstreams array is empty or lacks an application/pdf entry, the script must gracefully skip the item.