{% if referencefile %}
.. include:: {{ referencefile }}
{% endif %}

{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:

   {% block attributes_summary %}
   {% if attributes %}

   .. rubric:: Attributes

   .. autosummary::
      :toctree: ../stubs/
   {% for item in all_attributes %}
      {%- if not item.startswith('_') %}
      {{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods_summary %}
   {% if methods %}

   .. rubric:: Methods

   .. autosummary::
      :toctree: ../stubs/
   {% for item in all_methods %}
      {%- if not item.startswith('_') or item in ['__call__', '__mul__', '__add__', '__sub__', '__xor__', '__neg__', '__invert__', '__eq__', '__truediv__', '__matmul__', '__pow__', '__getitem__', '__len__'] %}
      {{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% for item in inherited_members %}
      {%- if not item.startswith('_') %}
      {{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}

   {% endif %}
   {% endblock %}
