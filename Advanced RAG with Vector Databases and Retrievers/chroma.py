import chromadb  # type: ignore
from chromadb.utils import embedding_functions  # type: ignore


ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.Client()

collection_name = "a_collection"


def main():
    try:
        Acollection = client.create_collection(
            name=collection_name,
            metadata={
                "description": "A test sample collection of chromadb created using create_collection method"
            },
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef,
            },
        )
        print(f"Collection created:\n{collection_name}")

        employees = [
            {
                "id": "employee_1",
                "name": "John Doe",
                "experience": 5,
                "department": "Engineering",
                "role": "Software Engineer",
                "skills": "Python, JavaScript, React, Node.js, databases",
                "location": "New York",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_2",
                "name": "Jane Smith",
                "experience": 8,
                "department": "Marketing",
                "role": "Marketing Manager",
                "skills": "Digital marketing, SEO, content strategy, analytics, social media",
                "location": "Los Angeles",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_3",
                "name": "Alice Johnson",
                "experience": 3,
                "department": "HR",
                "role": "HR Coordinator",
                "skills": "Recruitment, employee relations, HR policies, training programs",
                "location": "Chicago",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_4",
                "name": "Michael Brown",
                "experience": 12,
                "department": "Engineering",
                "role": "Senior Software Engineer",
                "skills": "Java, Spring Boot, microservices, cloud architecture, DevOps",
                "location": "San Francisco",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_5",
                "name": "Emily Wilson",
                "experience": 2,
                "department": "Marketing",
                "role": "Marketing Assistant",
                "skills": "Content creation, email marketing, market research, social media management",
                "location": "Austin",
                "employment_type": "Part-time",
            },
            {
                "id": "employee_6",
                "name": "David Lee",
                "experience": 15,
                "department": "Engineering",
                "role": "Engineering Manager",
                "skills": "Team leadership, project management, software architecture, mentoring",
                "location": "Seattle",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_7",
                "name": "Sarah Clark",
                "experience": 8,
                "department": "HR",
                "role": "HR Manager",
                "skills": "Performance management, compensation planning, policy development, conflict resolution",
                "location": "Boston",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_8",
                "name": "Chris Evans",
                "experience": 20,
                "department": "Engineering",
                "role": "Senior Architect",
                "skills": "System design, distributed systems, cloud platforms, technical strategy",
                "location": "New York",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_9",
                "name": "Jessica Taylor",
                "experience": 4,
                "department": "Marketing",
                "role": "Marketing Specialist",
                "skills": "Brand management, advertising campaigns, customer analytics, creative strategy",
                "location": "Miami",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_10",
                "name": "Alex Rodriguez",
                "experience": 18,
                "department": "Engineering",
                "role": "Lead Software Engineer",
                "skills": "Full-stack development, React, Python, machine learning, data science",
                "location": "Denver",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_11",
                "name": "Hannah White",
                "experience": 6,
                "department": "HR",
                "role": "HR Business Partner",
                "skills": "Strategic HR, organizational development, change management, employee engagement",
                "location": "Portland",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_12",
                "name": "Kevin Martinez",
                "experience": 10,
                "department": "Engineering",
                "role": "DevOps Engineer",
                "skills": "Docker, Kubernetes, AWS, CI/CD pipelines, infrastructure automation",
                "location": "Phoenix",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_13",
                "name": "Rachel Brown",
                "experience": 7,
                "department": "Marketing",
                "role": "Marketing Director",
                "skills": "Strategic marketing, team leadership, budget management, campaign optimization",
                "location": "Atlanta",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_14",
                "name": "Matthew Garcia",
                "experience": 3,
                "department": "Engineering",
                "role": "Junior Software Engineer",
                "skills": "JavaScript, HTML/CSS, basic backend development, learning frameworks",
                "location": "Dallas",
                "employment_type": "Full-time",
            },
            {
                "id": "employee_15",
                "name": "Olivia Moore",
                "experience": 12,
                "department": "Engineering",
                "role": "Principal Engineer",
                "skills": "Technical leadership, system architecture, performance optimization, mentoring",
                "location": "San Francisco",
                "employment_type": "Full-time",
            },
        ]

        print(f"Employees data created\n{employees}")

        employee_doc = []
        for employee in employees:
            document = (
                f"{employee['role']} with {employee['experience']} years of experience in {employee['department']}. "
            )
            document += f"Skills: {employee['skills']}. Located in {employee['location']}. "
            document += f"Employment type: {employee['employment_type']}."
            employee_doc.append(document)

        Acollection.add(
            ids=[e["id"] for e in employees],
            documents=employee_doc,
            metadatas=[
                {
                    "name": employee["name"],
                    "department": employee["department"],
                    "role": employee["role"],
                    "experience": employee["experience"],
                    "location": employee["location"],
                    "employment_type": employee["employment_type"],
                }
                for employee in employees
            ],
        )

        all_items = Acollection.get()
        print(f"Number of documents: {len(all_items['documents'])}")

        def safe_dist(value):
            try:
                return float(value)
            except Exception:
                return value

        def perform_advance_search(collection, _all_items):
            try:
                # -------------------------- 1  ---------------------------------
                print("=== Similarity Search Examples ===")
                query_text = "Python developer with web development experience"
                results = collection.query(query_texts=[query_text], n_results=3)

                if not results or not results.get("ids") or len(results["ids"][0]) == 0:
                    print("no result found")
                else:
                    for i, (doc_id, document, distance) in enumerate(
                        zip(results["ids"][0], results["documents"][0], results["distances"][0])
                    ):
                        metadata = results["metadatas"][0][i]
                        dist_val = safe_dist(distance)
                        # distance should be float, but handle strings/other types robustly
                        try:
                            dist_str = f"{dist_val:.4f}"
                        except Exception:
                            dist_str = str(dist_val)

                        print(f"  {i+1}. {metadata['name']} ({doc_id}) - Distance: {dist_str}")
                        print(
                            f"     Role: {metadata['role']}, Department: {metadata['department']}"
                        )
                        print(f"     Document: {document[:100]}...")

                # -------------------------- 2  ---------------------------------
                print("\n=== Metadata Filtering Examples ===")
                results = collection.get(where={"department": "Marketing"})

                if not results or not results.get("ids") or len(results["ids"][0]) == 0:
                    print("no result found")
                else:
                    for i, _doc_id in enumerate(results["ids"]):
                        metadata = results["metadatas"][i]
                        print(
                            f"  - {metadata['name']}: {metadata['role']} ({metadata['experience']} years)"
                        )

                # -------------------------- 3  ---------------------------------
                print("\n=== Combined Search: Similarity + Metadata Filtering ===")
                query_text3 = "senior Python developer full-stack"
                results = collection.query(
                    query_texts=[query_text3],
                    n_results=3,
                    where={
                        "$and": [
                            {"experience": {"$gte": 3}},
                            {"location": {"$in": ["San Francisco", "New York", "Seattle"]}},
                        ]
                    },
                )

                if not results or not results.get("ids") or len(results["ids"][0]) == 0:
                    print("no result found")
                else:
                    print(f"Query: '{query_text3}' with filters (3+ years, major tech cities)")
                    print(f"Found {len(results['ids'][0])} matching employees:")
                    for i, (doc_id, document, distance) in enumerate(
                        zip(results["ids"][0], results["documents"][0], results["distances"][0])
                    ):
                        metadata = results["metadatas"][0][i]
                        dist_val = safe_dist(distance)
                        try:
                            dist_str = f"{dist_val:.4f}"
                        except Exception:
                            dist_str = str(dist_val)

                        print(f"  {i+1}. {metadata['name']} ({doc_id}) - Distance: {dist_str}")
                        print(
                            f"     {metadata['role']} in {metadata['location']} ({metadata['experience']} years)"
                        )
                        print(f"     Document snippet: {document[:80]}...")

            except Exception as e:
                print(f"Error in perform_advance_search function\n{e}")

        perform_advance_search(Acollection, all_items)

    except Exception as e:
        print(f"Error encountered in main inside Chroma script\n{e}")


if __name__ == "__main__":
    main()

