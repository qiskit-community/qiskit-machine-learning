{% if referencefile %}
.. include:: {{ referencefile }}
{% endif %}

{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :no-members:
   :no-inherited-members:
   :no-special-members:

   {% block attributes_summary %}
   {% if attributes %}

   .. rubric:: Attributes

   {% for item in all_attributes %}
      {%- if item not in inherited_members %}
        {%- if not item.startswith('_') %}
   .. autoattribute:: {{ name }}.{{ item }}
        {%- endif -%}
      {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods_summary %}
   {% if methods %}

   .. rubric:: Methods

   {% for item in all_methods %}
      {%- if item not in inherited_members %}
        {%- if not item.startswith('_') %}
   .. automethod:: {{ name }}.{{ item }}
        {%- endif -%}
      {%- endif %}
   {%- endfor %}

   {% endif %}
   {% endblock %}
