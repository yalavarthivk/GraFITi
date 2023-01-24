{# PRE-CONFIGURATION #########################################################}
{% if obj.all is not none %}
   {% set visible_children = obj.children|selectattr("short_name", "in", obj.all)|list %}
{% elif obj.type is equalto("package") %}
   {% set visible_children = obj.children|selectattr("display")|list %}
{% else %}
   {% set visible_children = obj.children|selectattr("display")|rejectattr("imported")|list %}
{% endif %}

{% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}
{% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}
{% set visible_attributes = visible_children|selectattr("type", "equalto", "data")|list %}
{% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
{% set visible_submodules = obj.submodules|selectattr("display")|list %}

{% if not visible_subpackages or visible_submodules %}
:orphan:
{% endif %}


:py:mod:`{{ obj.name }}`
=========={{ "=" * obj.name|length }}

.. py:module:: {{ obj.name }}

{% if obj.docstring %}
.. autoapi-nested-parse::

   {{ obj.docstring|prepare_docstring|indent(3) }}
{% endif %}


{# SUB-PACKAGES ##############################################################}
{% block subpackages %}
{% if visible_subpackages %}
.. toctree::
   :titlesonly:
   :maxdepth: 3
   :hidden:

{% for subpackage in visible_subpackages %}
   {{ subpackage.short_name }}/index.rst
{% endfor %}

.. rubric:: Sub-Packages
.. autoapisummary::

{% for subpackage in visible_subpackages %}
   {{ subpackage.id }}
{% endfor %}
{% endif %}
{% endblock %}


{# SUB-MODULES ###############################################################}
{% block submodules %}
{% if visible_submodules %}
.. toctree::
   :titlesonly:
   :maxdepth: 3
   :hidden:

{% for submodule in visible_submodules %}
   {{ submodule.short_name }}/index.rst
{% endfor %}

.. rubric:: Sub-Modules
.. autoapisummary::

{% for submodule in visible_submodules %}
   {{ submodule.id }}
{% endfor %}
{% endif %}
{% endblock %}


{% block summary %}
{% if "show-module-summary" in autoapi_options and (visible_classes or visible_functions or visible_attributes) %}
{#{{ obj.type|title }} Summary#}
{#{{ "-" * obj.type|length }}---------#}


{# ATTRIBUTES ################################################################}
{% block attributes scoped %}
{% if visible_attributes %}
.. rubric:: Attributes
.. autoapisummary::

{% for attribute in visible_attributes %}
   {{ attribute.id }}
{% endfor %}
{% endif %}
{% endblock %}


{# CLASSES ###################################################################}
{% block classes scoped %}
{% if visible_classes %}
.. rubric:: Classes
.. autoapisummary::

{% for klass in visible_classes %}
   {{ klass.id }}
{% endfor %}

{% endif %}
{% endblock %}


{# FUNCTIONS #################################################################}
{% block functions scoped %}
{% if visible_functions %}
{#.. toctree::#}
{#   :titlesonly:#}
{#   :maxdepth: 3#}
{#   :hidden:#}
{##}
{#{% for function in visible_functions %}#}
{#   {{ function.short_name }}/index.rst#}
{#{% endfor %}#}

.. rubric:: Functions
.. autoapisummary::

{% for function in visible_functions %}
   {{ function.id }}
{% endfor %}
{% endif %}
{% endblock %}

{% endif %}
{% endblock %}


{# CONTENT ###################################################################}
{% block content %}
{% if visible_children %}
{#{{ obj.type|title }} Contents#}
{#{{ "-" * obj.type|length }}---------#}

{% for obj_item in visible_children %}
{{ obj_item.render()|indent(0) }}
{% endfor %}

{% endif %}
{% endblock %}
