import requests
import re
import os
import base64
from PIL import Image
import ollama
from Flask import Flask, render_template, request, redirect, flash, url_for

