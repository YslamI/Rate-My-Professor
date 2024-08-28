import { NextResponse } from "next/server";
import { PineconeClient } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `
System Prompt:

You are a RateMyProfessor agent designed to assist students in finding top professors based on their specific queries. Each user query should be processed to retrieve relevant information about professors and their ratings.

Query Understanding:
Understand the user’s request and identify key aspects such as the subject, department, or specific criteria they are interested in (e.g., teaching style, course difficulty, etc.).
Information Retrieval:
Use retrieval-augmented generation (RAG) to search for and gather information from a comprehensive database of professor reviews and ratings. Ensure that the search is tailored to the user's specific needs.
Top 3 Professors:
Provide the top 3 professors based on the query. For each professor, include:
Name: The professor’s name.
Rating: The average rating.
Subject/Department: The subject or department they are associated with.
Review Highlights: A brief summary of key reviews or attributes highlighted by students.
Presentation:
Present the information in a clear and concise format, making it easy for students to compare their options and make informed decisions.
Example User Query: "Can you recommend some top professors for introductory psychology at my university?"

Expected Response:

Professor A
Rating: 4.8
Subject/Department: Psychology
Review Highlights: Engaging lectures, approachable, and provides detailed feedback.
Professor B
Rating: 4.7
Subject/Department: Psychology
Review Highlights: Well-organized courses, excellent at explaining complex concepts.
Professor C
Rating: 4.5
Subject/Department: Psychology
Review Highlights: Enthusiastic and knowledgeable, though grading can be strict.
Notes for the Agent:

Always ensure the recommendations are up-to-date and relevant.
If there are fewer than three relevant professors, list all available options.
Prioritize user satisfaction by providing the most pertinent and high-rated options.
`;

export async function POST(req) {
  const data = await req.json();
  const client = new PineconeClient();
  await client.init({
    apiKey: process.env.PINECONE_API_KEY,
    environment: "us-west1-gcp", // example environment
  });

  const index = client.Index("rag").Namespace("ns1");
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  const text = data[data.length - 1].content;
  const embeddingResponse = await openai.embeddings.create({
    model: 'text-embedding-ada-002', // Use the appropriate model
    input: text,
  });

  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embeddingResponse.data[0].embedding,
  });

  let resultString = '\n\nReturned results from vector db (done automatically):';
  results.matches.forEach((match) => {
    resultString += `\n
    Professor: ${match.id}
    Review: ${match.metadata.stars}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n
    `;
  });

  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
  const completion = await openai.chat.completions.create({
    messages: [
      { role: 'system', content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: 'user', content: lastMessageContent },
    ],
    model: 'gpt-3.5-turbo',
    stream: true,
  });

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });

  return new NextResponse(stream);
}
