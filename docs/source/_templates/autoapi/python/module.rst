{{ obj.name }}
{{ "=" * obj.name|length }}

.. py:module:: {{ obj.name }}

{# Absolute base docname for this module, e.g. /autoapi/baccmod/bkg_collection #}
{% set base = "/autoapi/" + obj.name.replace(".", "/") %}

{# Submodules (if any) #}
.. toctree::
   :maxdepth: 1
   :glob:
   :titlesonly:

   {{ base }}/*/index

{# Classes defined in this module (AutoAPI writes class pages as PascalCase docs) #}
{% if "class" in own_page_types %}
Classes
-------
.. toctree::
   :maxdepth: 1
   :glob:
   :titlesonly:

   {{ base }}/[A-Z]*
{% endif %}

{% if obj.docstring %}
{{ obj.docstring }}
{% endif %}
